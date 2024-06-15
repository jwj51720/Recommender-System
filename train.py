import argparse
import torch
from sources.utils import *
from sources.trainer import Trainer
from sources.data_loader import *


def main(args):
    print("This script is running directly")
    model = get_model(args)
    print(model)
    train_loader, valid_loader = get_loader(args.config)
    trainer = Trainer(args, model, train_loader, valid_loader)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.json",
        help="Path to the training configuration file",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    with open(args.config, "r") as f:
        args.config = json.load(f)
    main(args)
