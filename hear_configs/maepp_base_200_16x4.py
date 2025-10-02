import os
import sys
from hear_api.runtime import RuntimeMAE
import ml_collections
from importlib import import_module
PT_MAEPP_MODEL_DIR = os.environ.get("PT_MAEPP_MODEL_DIR")


config_path = "configs.maepp_base_200_16x4"
precision = "float16"
model_path = os.path.join(PT_MAEPP_MODEL_DIR, f"maepp_base_200_16x4")


def load_model(model_path=model_path, config=import_module(config_path).get_config()):
    model = RuntimeMAE(config=config, weights_dir=model_path, precision=precision)
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)