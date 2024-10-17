from typing import Dict, Optional, Tuple, List
import random
import logging
import os
import shutil
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)
from torch.optim.lr_scheduler import (
    StepLR,
)
from torch.optim import Adam, AdamW
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import yaml
import torch
import numpy as np


def load_config(config_path: str) -> Dict:
    """Load the configuration from a given path and return a dictionary with the parameters.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Dictionary with configuration parameters.
    """
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise IOError(f"Failed to load configuration from {config_path}: {e}")


def init_logging(
    log_dir: Optional[str] = None, log_level: int = logging.INFO
) -> logging.Logger:
    """Initialize logging with a timestamped log file.

    Args:
        log_dir (Optional[str]): Directory to store log files. Defaults to 'logs'.
        log_level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = log_dir if log_dir else "logs"
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except Exception as e:
            raise IOError(f"Failed to create log directory {log_dir}: {e}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{log_dir}/log_{timestamp}.log"

    logging.basicConfig(
        filename=log_filename,
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Initialized logging.")
    return logger


def load_and_log_config(config_path: str) -> Tuple[Dict, logging.Logger]:
    """
    Load the config file and initialize logger.

    Parameters:
        config_path (str): Path to config file.

    Returns:
        tuple: Tuple containing loaded config (dict) and initialized logger.
    """
    # Load config and initialize logger
    config = load_config(config_path)
    logger = init_logging()
    return config, logger


def load_model_and_tokenizer(
    model_name: str,
    device: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a pretrained model and tokenizer from HuggingFace.

    Args:
        model_name (str): Name of the model to load.
        device (str): Device to use for the model.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


def get_optimizer(model: PreTrainedModel, config: Dict) -> torch.optim.Optimizer:
    """Get the optimizer for the model.

    Args:
        model (PreTrainedModel): The language model.
        config (Dict): Configuration parameters.

    Raises:
        ValueError: If an invalid optimizer is specified in the config file.

    Returns:
        torch.optim.Optimizer: Optimizer for training.
    """
    optimizer_type = config["training_kwargs"].get("optimizer", "adamw_torch")

    weight_decay = config["training_kwargs"].get("weight_decay", 0.01)
    betas = config["training_kwargs"].get("betas", (0.9, 0.999))
    eps = config["training_kwargs"].get("eps", 1e-8)

    if optimizer_type == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=config["training_kwargs"]["learning_rate"],
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "adamw_torch":
        optimizer = AdamW(
            model.parameters(),
            lr=config["training_kwargs"]["learning_rate"],
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=False,
        )
    else:
        raise ValueError("Invalid optimizer specified in config file.")
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict, steps: int
) -> torch.optim.lr_scheduler.StepLR:
    """Get the learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer for training.
        config (Dict): Configuration parameters.

    Raises:
        ValueError: If an invalid lr_scheduler_type is specified in the config file.

    Returns:
        torch.optim.lr_scheduler.StepLR: _description_
    """
    if config["training_kwargs"].get("lr_scheduler_type", "step_lr") == "step_lr":
        scheduler = StepLR(
            optimizer,
            step_size=config["training_kwargs"].get("lr_step_size", 10),
            gamma=config["training_kwargs"].get("lr_gamma", 0.1),
        )
    elif config["training_kwargs"].get("lr_scheduler_type", "step_lr") == "cosine":
        # scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            int(steps / 10),
            cycle_mult=config["training_kwargs"].get("cycle_mult", 1.8),
            max_lr=config["training_kwargs"].get("learning_rate", 0.00002),
            min_lr=config["training_kwargs"].get("min_lr", 0.0000002),
            warmup_steps=config["training_kwargs"].get("warmup_steps", 100),
            gamma=config["training_kwargs"].get("gamma", 0.8),
        )

    else:
        raise ValueError("Invalid lr_scheduler_type specified in config file.")
    return scheduler


def set_global_seed(seed: int):
    """Set the seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        ":4096:8"  # required for deterministic training
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: PreTrainedModel,
    logger: logging.Logger,
    saved_checkpoints: List[str],
    step: str,
    eval_loss: float,
    save_path: str,
    max_checkpoints: int = 2,
):
    checkpoint_path = f"{save_path}/model_step_{step}"
    model.save_pretrained(checkpoint_path)
    logger.info(f"Saved best model with loss: {eval_loss:.4f}")

    saved_checkpoints.append(checkpoint_path)

    # check if delete the oldest checkpoint
    if len(saved_checkpoints) > max_checkpoints:
        oldest_checkpoint = saved_checkpoints.pop(0)
        shutil.rmtree(oldest_checkpoint)
        logger.info(f"Deleted oldest checkpoint: {oldest_checkpoint}")
