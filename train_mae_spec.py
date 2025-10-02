import glob
import random
import numpy as np
import torch
import time
import math
import json
import tqdm
import sys
import functools
import os
from pathlib import Path
from importlib import import_module
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from absl import logging
import wandb
import ml_collections
from src.loss import mae_loss
from src import utilities
from src import misc
from src import lr_sched
from src.misc import NativeScalerWithGradNormCount as NativeScaler
import argparse
import timm.optim.optim_factory as optim_factory
from src.mae_engine import train_mae
from src.setup_data import setup_data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributed.elastic.multiprocessing.errors import record
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=True)
    parser.add_argument("--config", default="", type=str, help="path to config file")
    parser.add_argument('--workdir', default='', type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--fake_data', action='store_true')
    parser.add_argument('--bias_decay', action='store_true')
    parser.add_argument('--use_rs', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--precision", default='float32', type=str)
    parser.add_argument("--min_lr", type=float, default=0.)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


@record
def main(args):
    config = args.config
    misc.init_distributed_mode(args)
    print("WORLD_SIZE:", args.world_size)
    print("LOCAL_RANK:", args.gpu)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("num_tasks:", num_tasks)
    print("global_rank:", global_rank)

    workdir = args.workdir

    args.output_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(args.output_dir, exist_ok=True)
    
    wandb_logger = None
    if misc.is_main_process() and not args.no_wandb:
        wandb_logger = wandb.init(project='{}'.format(config.wandb.get("project", "audax-cola")),
                                  group="{}".format(config.data.dataset_name),
                                  config=config.to_dict(), name=workdir.split("/")[-1])
    else:
        wandb_logger = None

    device = torch.device("cuda:{}".format(args.gpu))

    train_iter, steps_per_epoch, samples = setup_data(config, train=True, num_workers=args.num_workers, use_reading_service=args.use_rs)
    if args.fake_data:
        print("USING FAKE DATA")
        tr_set = datasets.FakeData(samples, (1, 200, 80), 1000, transforms.ToTensor())
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(tr_set)
        train_iter = torch.utils.data.DataLoader(
            tr_set, batch_size=config.batch_size, shuffle=(train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    num_tr_steps = int(steps_per_epoch * config.num_epochs)

    args.steps_per_epoch = steps_per_epoch
    args.warmup_epochs = config.opt.warmup_epochs
    args.epochs = config.num_epochs

    print("Total steps: {} | Steps per epoch: {}".format(num_tr_steps, args.steps_per_epoch))

    # create model here

    model = utilities.get_model(config)
    model.to(device)
    model_without_ddp = model
    print(model)

    if args.distributed:
        print("DISTRIBUTED IS TRUEEEEEEEE...")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # create criterion here
    use_norm_pix = config.opt.get("norm_pix_loss", False)
    criterion = functools.partial(mae_loss, norm_pix_loss=use_norm_pix)

    # create optimizer here
    args.accum_iter = config.opt.get("grad_accum_steps", 1)
    args.clip_grad_value = config.opt.get("clip_grad_value", None)
    base_learning_rate = args.accum_iter * config.opt.learning_rate * config.batch_size * args.world_size / 256.
    wd = config.opt.weight_decay
    args.lr = base_learning_rate
    args.global_bs = config.batch_size * args.world_size    

    if args.bias_decay:
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=wd)
    else:
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, wd)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        
    print(optimizer)
    loss_scaler = NativeScaler()

    final_epochs = list(range(config.num_epochs))[-5:]
    if args.resume:
       misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    elif os.path.exists(args.output_dir):
       func = lambda x:int(x.split("/")[-1].split("-")[-1].replace(".pth",""))
       existing_ckpts = sorted(glob.glob(os.path.join(args.output_dir, "*.pth")), key=func)
       if len(existing_ckpts) != 0:
           latest_ckpt = existing_ckpts[-1]
           args.resume = latest_ckpt
           misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    total_steps_counter = 0
    for epoch in range(args.start_epoch, config.num_epochs):
        train_stats, total_steps_counter = train_mae(
            train_iter, model,
            criterion, optimizer, epoch, steps_per_epoch, device,
            loss_scaler, args, wandb_logger,
            total_steps_counter
        )

        if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs or epoch in final_epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.workdir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    misc.barrier()
    if wandb_logger is not None:
        wandb_logger.finish()
    misc.cleanup()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    config = import_module(args.config).get_config()
    args.config = config
    if args.workdir:
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
    main(args)
