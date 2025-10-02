import torch
import functools
from src.data import dataset_helper, parsing_utilities
from torchdata.datapipes.iter import FileLister, FileOpener, Shuffler
from torchdata.dataloader2 import DataLoader2, DistributedReadingService, MultiProcessingReadingService, SequentialReadingService


def setup_data(config, train=False, num_workers=4, use_reading_service=False):
    _, samples = dataset_helper.get_data_dirs(config.data.train_dirs, config.data.train_samples)
    frequency_first = config.model.model_args.get("frequency_first", False)
    parser_fn = functools.partial(
            parsing_utilities.np_spec_parser, 
            req_num_frames=config.data.num_frames,
            crop_type="random" if train else "center",
            flip_ft=frequency_first
        )
    record_parser_fn = functools.partial(
            parsing_utilities.numpy_record_parser,
            numpy_spec_parser_fn=parser_fn,
        )
    shuffle_buffer = config.get("shuffle_buffer_multiplier", 1000)
    eff_shuffle_buffer = ((shuffle_buffer*config.batch_size*10)//(num_workers*torch.distributed.get_world_size()))
    print("Effective shuffle buffer:", eff_shuffle_buffer)
    dp = FileLister(config.data.train_dirs, "*.tar")
    dp = Shuffler(dp, buffer_size=1000)
    dp = dp.sharding_filter()
    dp = FileOpener(dp, mode='b')
    dp = dp.load_from_tar(length=samples).map(dataset_helper.decode_np_tar).webdataset()
    dp = Shuffler(dp, buffer_size=eff_shuffle_buffer)
    dp = dp.map(dataset_helper.fix_keys_for_tp).map(record_parser_fn).map(dataset_helper.return_data)
    dp = dp.cycle()

    if use_reading_service:
       print("!!!!!!!!!! using reading service !!!!!!!!!!!")
       dp = dp.batch(config.batch_size, drop_last=False).prefetch(config.batch_size*20)
       dp = dp.collate()
       mp_rs = MultiProcessingReadingService(num_workers=num_workers)
       dist_rs = DistributedReadingService()
       rs = SequentialReadingService(dist_rs, mp_rs)
       loader = DataLoader2(dp, reading_service=rs)
    else:
       loader = torch.utils.data.DataLoader(
           dp, batch_size=config.batch_size, shuffle=True,
           num_workers=num_workers, drop_last=False, pin_memory=False
       )

    return loader, samples//(config.batch_size * torch.distributed.get_world_size()), samples