import sys
import math
import time
import torch
from . import lr_sched, utilities, misc


def train_mae(train_iter, model,
          criterion, optimizer,
          epoch, steps_per_epoch,
          device,
          loss_scaler,
          args,
          wandb_logger=None,
          total_steps_counter=0,
          ):
    model.train()
    prefetcher = utilities.Prefetcher(train_iter, device)
    inp, _ = prefetcher.next()
    accum_iter = args.accum_iter
    autocast_dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    autocast_enabled = True if "16" in args.precision else False
    step_times = []

    loss_values = []

    data_iter_step = 0
    #with model.join():
        # while inp is not None:
    for step_index in range(steps_per_epoch):
            t0 = time.time()
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / args.steps_per_epoch + epoch, args)
            # print("now forwarding..")
            if step_index == 0:
               print("inp shape:", inp.shape)
            with torch.autocast(device_type="cuda", 
                                dtype=autocast_dtype,
                                enabled=autocast_enabled):
                pred, target, mask = model(inp)
                loss = criterion(pred, target, mask)
            loss = loss.mean()
            loss_value = loss.item()
            # print("got loss")

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= accum_iter
            cond = (data_iter_step + 1) % accum_iter == 0
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad_value, 
                                    parameters=model.parameters(),
                                    update_grad=cond)
            if cond:
                optimizer.zero_grad()
        
            # torch.cuda.synchronize()
            # print("grad_norm: {} | loss: {}".format(grad_norm, loss))
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            grad_norm_reduce = misc.all_reduce_mean(grad_norm)
            # print("loss value reduced:", loss_value_reduce)
            lr = optimizer.param_groups[0]["lr"]
            step_times.append(time.time()-t0)
            steps_per_sec = 1 / (sum(step_times[-5:]) / len(step_times[-5:]))
            samples_per_second = args.global_bs * steps_per_sec
            if data_iter_step % args.print_freq == 0:
                print("Epoch: {:03d} [{:04d}] | loss: {:.04f} | grad_norm: {:.04f} | steps_per_second: {:.02f} | samples_per_second: {:.02f} | lr: {:.08f}".format(
                    epoch, data_iter_step, loss_value_reduce, grad_norm_reduce, steps_per_sec, samples_per_second, lr)
                )
            if wandb_logger is not None and misc.is_main_process():
                wandb_logger.log({
                    "train_reconstruction_loss": loss_value_reduce,
                    "step": total_steps_counter,
                    "lr": lr,
                    "steps_per_second": steps_per_sec,
                    "samples_per_second": samples_per_second,
                    "grad_norm": grad_norm_reduce
                })
            
            loss_values.append(loss_value_reduce)
            data_iter_step += 1
            total_steps_counter += 1
            inp, _ = prefetcher.next()
    
    mean_loss = sum(loss_values)/len(loss_values)
    print("Epoch: {} | Mean tr loss: {}".format(epoch, mean_loss))
    return {"loss": mean_loss}, total_steps_counter
