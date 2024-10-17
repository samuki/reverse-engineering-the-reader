from typing import Dict
import os

import wandb
from models.utils.secret_credentials import WANDB_API_KEY, WANDB_LOG_MODEL


def init_wandb(config: Dict, project_name: str, resume: bool = False, id: str = None):
    """
    Initialize wandb for logging.
    """
    save_scratch = config.get("save_scratch", False)
    if save_scratch:
        from models.utils.secret_credentials import username

        default_cache_data_dir = f"/cluster/scratch/{username}/wandb"
    else:
        default_cache_data_dir = "wandb_data"

    seed = config["seed"]
    model = config["model"]
    if model.endswith("gpt2"):
        model_name = "GPT2-S"
    elif model.endswith("gpt2-medium"):
        model_name = "GPT2-M"
    elif model.endswith("gpt2-large"):
        model_name = "GPT2-L"

    preference_dataset = config["training_kwargs"]["preference_dataset"]
    eval_dataset = config["training_kwargs"]["eval_dataset"]
    if eval_dataset == "preference_dataset":
        eval_dataset = preference_dataset
    dataset_mapping = {
        "dundee": "D",
        "provo": "P",
        "zuco": "Z",
    }
    run_name = f"{model_name}: {dataset_mapping[preference_dataset]} ➛ {dataset_mapping[eval_dataset]}_{seed}"
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_LOG_MODEL"] = WANDB_LOG_MODEL
    os.environ["WANDB_DATA_DIR"] = default_cache_data_dir
    os.environ["WANDB_CACHE_DIR"] = default_cache_data_dir
    if resume:
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = id
    wandb.init(project=project_name, config=config, name=run_name)


def log_wandb(
    step: int,
    train_loss: float,
    eval_loss: float,
    lr: float,
    ce_loss: float,
    ppl_ce_all: float,
    coefficients_llm: Dict[str, float],
    ppl_mean: float,
    delta_llh: float,
    mse_base: float,
    mse_trg: float,
    coefficients_base: Dict[str, float],
    coefficients_trg: Dict[str, float],
    llh_base: float,
    llh_trg: float,
    kl_mean: float,
    kl_weight: float,
    train_kls_mean: float,
):
    """Log metrics to wandb."""
    all_metrics = {
        "train/step": step,
        "train/train_loss": train_loss,
        "train/lr": lr,
        "train/train_kl": train_kls_mean,
        "eval/eval_loss": eval_loss,
        "eval/ppl_mean": ppl_mean,
        "eval/ppl_ce_all": ppl_ce_all,
        "eval/ce_loss": ce_loss,
        "eval/kl_mean": kl_mean,
        "eval/kl_weight": kl_weight,
        "linear_regression/delta_llh": delta_llh,
        "linear_regression/mse_base": mse_base,
        "linear_regression/mse_trg": mse_trg,
        "linear_regression/llh_base": llh_base,
        "linear_regression/llh_trg": llh_trg,
    }
    for key, value in coefficients_llm.items():
        all_metrics[f"llm_coefficients/{key}"] = value
    for key, value in coefficients_base.items():
        all_metrics[f"linear_regression_coefficients_base/{key}"] = value
    for key, value in coefficients_trg.items():
        all_metrics[f"linear_regression_coefficients_trg/{key}"] = value
    wandb.log(all_metrics)
