import torch
import argparse

from models.train import train
from models.utils.utils import (
    load_model_and_tokenizer,
    set_global_seed,
    load_and_log_config,
)
from data.load_data import load_data


def main(config_path):
    config, logger = load_and_log_config(config_path)
    logger.info(f"Training parameters loaded: {config}")

    # Set all global seeds
    set_global_seed(config["seed"])

    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        from models.utils.wandb_utils import init_wandb

        project_name = config.get("wandb_project_name", "lrdpo")
        resume = config.get("resume", False)
        run_id = config.get("run_id", "")
        init_wandb(config, project_name, resume=resume, id=run_id)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model_name = config["model"]
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model and tokenizer loaded: {model_name}")

    train_loader, eval_loader, mean_rt_train, std_rt_train = load_data(
        config, tokenizer
    )

    if config["training_kwargs"].get("kl_variant", None) == "full":
        model_ref, _ = load_model_and_tokenizer(model_name, device)
    else:
        model_ref = None
    train(
        model,
        tokenizer,
        train_loader,
        eval_loader,
        config,
        logger,
        device,
        model_ref=model_ref,
        mean_rt_train=mean_rt_train,
        std_rt_train=std_rt_train,
    )


if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="LRDPO")
    parser.add_argument(
        "--config_path", type=str, help="Path to the configuration file"
    )
    args = parser.parse_args()

    main(args.config_path)
