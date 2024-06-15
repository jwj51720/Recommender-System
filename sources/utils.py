import json
import os
from models import FactorizationMachine
import torch.nn as nn
import torch
from torch.optim import Adam


def get_model(args):
    model_name = args.config["model"]
    if model_name == "fm":
        model_config = get_hyperparameter(model_name)
        model = FactorizationMachine(**model_config)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model.to(args.device)


def get_hyperparameter(model_name):
    base_dir = os.path.dirname(__file__)  # 현재 파일의 디렉토리
    config_paths = {"fm": os.path.join(base_dir, "..", "configs", "fm.json")}

    if model_name not in config_paths:
        raise ValueError(f"Unknown model: {model_name}")

    config_file = config_paths[model_name]
    with open(config_file, "r") as f:
        config = json.load(f)

    return config


def get_criterion(config):
    if config["criterion"].lower().strip() == "mse":
        return nn.MSELoss()
    elif config["criterion"].lower().strip() == "crossentropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown criterion: {config['criterion']}")


def get_optimizer(config, model):
    if config["optimizer"] == "Adam":
        return Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
