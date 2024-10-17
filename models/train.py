from typing import Dict, Callable
from logging import Logger
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
import pandas as pd

from models.regression import fit_regression_models

from models.utils.wandb_utils import log_wandb
from models.utils.utils import (
    get_optimizer,
    get_scheduler,
    save_checkpoint,
)
from models.loss import compute_loss


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_data_loader: DataLoader,
    eval_data_loader: DataLoader,
    config: Dict,
    logger: Logger,
    device: torch.device,
    model_ref: PreTrainedModel = None,
    mean_rt_train: float = None,
    std_rt_train: float = None,
) -> None:
    """
    Train the model

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        train_data_loader (DataLoader): DataLoader for the training data.
        eval_data_loader (DataLoader): DataLoader for the evaluation data.
        condif (Dict): Loaded config.
        logger (Logger): Logger for logging.
        device (torch.device): The device to run the model on.
        model_ref (PreTrainedModel): Reference model for KL divergence.
        mean_rt_train (float): Mean of the training rts for random rts.
        std_rt_train (float): Std of the training rts for random rts.
    """

    # Setup all training configurations
    output_dir = config["training_kwargs"].get("output_dir", "output")
    evaluation_strategy = config["training_kwargs"].get("evaluation_strategy", "steps")
    eval_steps = config["training_kwargs"].get("eval_steps", 100)
    use_wandb = config.get("use_wandb", False)
    experiment_name = config["experiment_name"]
    reduction = config["training_kwargs"].get("reduction", "mean")
    save_scratch = config.get("save_scratch", False)
    accumulation_steps = config["training_kwargs"].get("accumulation_steps", 1)
    max_checkpoints = config["training_kwargs"].get("max_checkpoints", 1)
    saved_checkpoints = []
    mask_zero_reading_times = config["training_kwargs"].get(
        "mask_zero_reading_times", True
    )
    kl_weight = config["training_kwargs"].get("kl_weight", 0)
    kl_variant = config["training_kwargs"].get("kl_variant", None)

    mask_zero_freqs = config["training_kwargs"].get("mask_zero_freqs", True)
    total_steps_mutliplier = config["training_kwargs"]["total_steps_mutliplier"]
    train_w_random_rts = config["training_kwargs"].get("train_w_random_rts", False)
    seed = config.get("seed", 42)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, total_steps_mutliplier * eval_steps)
    loss = config["training_kwargs"].get("loss", "delta_llh_wt")
    loss_fn = compute_loss if loss == "delta_llh_wt" else None
    loss_type = config["training_kwargs"].get("loss_type", "mse")
    save_metric = config["training_kwargs"].get("save_metric", "loss")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/{experiment_name}_{seed}", exist_ok=True)

    step = 0
    best_eval_loss = float("inf")
    train_losses = []
    train_kls = []
    while step < total_steps_mutliplier * eval_steps:
        for b_index, batch in enumerate(tqdm(train_data_loader)):
            model.train()
            if model_ref:
                model_ref.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            train_loss, _, _, _, _, train_kl = loss_fn(
                model,
                tokenizer,
                batch,
                loss_type=loss_type,
                reduction=reduction,
                mask_zero_reading_times=mask_zero_reading_times,
                mask_zero_freqs=mask_zero_freqs,
                train_w_random_rts=train_w_random_rts,
                kl_weight=kl_weight,
                kl_variant=kl_variant,
                model_ref=model_ref,
                mean_rt_train=mean_rt_train,
                std_rt_train=std_rt_train,
            )
            train_losses.append(train_loss.item())
            train_kls.append(train_kl.item())

            logger.info(f"Step: {step}, Loss: {train_loss.item():.4f}")
            if evaluation_strategy == "steps" and step % eval_steps == 0:
                metrics = validate(
                    model,
                    tokenizer,
                    eval_data_loader,
                    device,
                    loss_type=loss_type,
                    loss_fn=loss_fn,
                    reduction=reduction,
                    mask_zero_reading_times=mask_zero_reading_times,
                    mask_zero_freqs=mask_zero_freqs,
                    kl_weight=kl_weight,
                    kl_variant=kl_variant,
                    model_ref=model_ref,
                )
                if save_metric == "loss":
                    eval_loss = metrics["loss"]
                else:
                    raise ValueError("Invalid save metric")
                if (
                    eval_loss < best_eval_loss
                    and step <= total_steps_mutliplier * eval_steps
                ):
                    best_eval_loss = eval_loss
                    if not save_scratch:
                        save_path = f"{output_dir}/{experiment_name}_{seed}/"
                    else:
                        from models.utils.credentials import username

                        save_path = f"/cluster/scratch/{username}/{output_dir}/{experiment_name}_{seed}/"

                    save_checkpoint(
                        model,
                        logger,
                        saved_checkpoints,
                        step,
                        eval_loss,
                        save_path,
                        max_checkpoints,
                    )
                if use_wandb:
                    log_wandb(
                        step,
                        np.mean(train_losses) if step > 0 else float("nan"),
                        metrics["loss"],
                        lr=scheduler.get_lr()[0],
                        ce_loss=metrics["ce_loss_all"].mean().item(),
                        ppl_ce_all=metrics["ppl_ce_all"].mean().item(),
                        coefficients_llm=metrics["coefficients"],
                        ppl_mean=metrics["ppl_values"].mean().item(),
                        delta_llh=metrics["delta_llh"],
                        mse_base=metrics["mse_base"],
                        mse_trg=metrics["mse_trg"],
                        coefficients_base=metrics["coefficients_base"],
                        coefficients_trg=metrics["coefficients_targ"],
                        llh_base=np.mean(metrics["llh_base"]),
                        llh_trg=np.mean(metrics["llh_trg"]),
                        kl_mean=np.mean(metrics["kl"]),
                        kl_weight=np.mean(metrics["kl_weight"]),
                        train_kls_mean=np.mean(train_kls),
                    )
                train_losses.clear()
            train_loss.backward()
            if (b_index + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                train_kls.clear()

            step += 1


def validate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data_loader: DataLoader,
    device: torch.device,
    loss_type: str = "mse",
    loss_fn: Callable = compute_loss,
    reduction: str = "mean",
    mask_zero_reading_times: bool = True,
    mask_zero_freqs: bool = True,
    kl_weight: float = 0.0,
    kl_variant: str = "abs",
    model_ref: PreTrainedModel = None,
) -> Dict:
    """
    Evaluate model on the evaluation data

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        eval_data_loader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): The device to run the model on.
        loss_type (str): The type of loss to compute. Defaults to "mse".
        loss_fn (Callable): The loss function to use. Defaults to compute_loss.
        reduction (str): The reduction type for the loss. Defaults to "mean". Alternative is "sum".
        mask_zero_reading_times (bool): Whether to mask zero reading times. Defaults to False.
        mask_zero_freqs (bool): Whether to mask zero frequencies. Defaults to False.
        kl_weight (float): The weight for the KL divergence. Defaults to 0.0.
        kl_variant (str): The variant of the KL divergence. Defaults to "abs".
        model_ref (PreTrainedModel): Reference model for KL divergence. Defaults to None.

    Returns:
        metrics dict (Dict): Metrics for logging
    """

    model.eval()
    if model_ref:
        model_ref.eval()
    total_loss = 0.0
    coefficients_collected = {}
    ppl_list = []
    ce_loss_list = []
    sentence_metrics_list = []
    all_kl = []
    all_kl_weight = []
    with torch.no_grad():
        for batch in tqdm(eval_data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, coefficients_all, ppl, ce_loss, sentence_metrics, kl = loss_fn(
                model,
                tokenizer,
                batch,
                loss_type=loss_type,
                reduction=reduction,
                mask_zero_reading_times=mask_zero_reading_times,
                mask_zero_freqs=mask_zero_freqs,
                is_eval=True,
                train_w_random_rts=False,
                kl_weight=kl_weight,
                kl_variant=kl_variant,
                model_ref=model_ref,
            )
            total_loss += loss.item()
            for key, value in coefficients_all.items():
                if key not in coefficients_collected:
                    coefficients_collected[key] = []
                coefficients_collected[key].append(value)
            ppl_list.append(ppl.unsqueeze(0))
            ce_loss_list.append(ce_loss.unsqueeze(0))
            sentence_metrics_list.append(sentence_metrics)
            all_kl.append(kl.cpu())
            all_kl_weight.append(kl_weight)

    ppl_all = torch.cat(ppl_list, dim=0)
    ce_loss_all = torch.cat(ce_loss_list, dim=0)
    ppl_ce_all = torch.exp(ce_loss_all)

    sentence_metrics_all = pd.concat(sentence_metrics_list, ignore_index=True)

    # Get Baseline predictors
    X1 = sentence_metrics_all[["word_length", "word_frequency"]]

    # Reading times
    y = sentence_metrics_all["reading_times"]

    # Get Target predictors including surprisal
    X2 = X1.copy()
    X2["surprisal"] = sentence_metrics_all["surprisal"]

    coefficients_base, llh_base, squared_errors_base = fit_regression_models(X1, y)
    coefficients_targ, llh_trg, squared_errors_target = fit_regression_models(X2, y)

    delta_llh = np.mean(llh_trg - llh_base)
    mean_squared_error_base = np.mean(squared_errors_base)
    mean_squared_error_target = np.mean(squared_errors_target)

    coefficients_collected = {
        k: torch.mean(torch.stack(v)).item() for k, v in coefficients_collected.items()
    }
    return {
        "loss": total_loss / len(eval_data_loader),
        "coefficients": coefficients_collected,
        "delta_llh": delta_llh,
        "ppl_values": ppl_all,
        "ce_loss_all": ce_loss_all,
        "ppl_ce_all": ppl_ce_all,
        "mse_base": mean_squared_error_base,
        "coefficients_base": coefficients_base,
        "llh_base": llh_base,
        "mse_trg": mean_squared_error_target,
        "coefficients_targ": coefficients_targ,
        "llh_trg": llh_trg,
        "kl": all_kl,
        "kl_weight": all_kl_weight,
    }
