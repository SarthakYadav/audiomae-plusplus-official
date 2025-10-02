import ml_collections
import os
from .common import get_opt_config, get_data_config
DATASET_DIR = os.environ.get("DATASETS_BASE_DIR", "")


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "mae_plusplus_large"
    config.model.type = "mae"
    config.model.num_classes = 1000     # needed for dataset parsing. Is not used.
    config.model.model_args = {
        "mask_ratio": 0.8,
        "img_size": (200, 80),
        "patch_size": (4, 16),
        "decoder_num_heads": 8,
        "decoder_depth": 4,
        "decoder_embed_dim": 512,
        "encoder_plusplus_block": True,
        "decoder_plusplus_block": True,
        "encoder_use_swiglu_final_ffn": True,
        "decoder_use_swiglu_final_ffn": True,
        "encoder_use_rope": False,
        "decoder_use_rope": False
    }
    config.model.patch_embed_args = ml_collections.ConfigDict()

    config.opt = get_opt_config()
    config.data = get_data_config()

    config.log_every_steps = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.batch_size = 128     # per gpu
    config.shuffle_buffer_multiplier = 500
    config.half_precision = False
    config.num_epochs = 100

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "mae-plusplus-pytorch"

    return config
