"""
Loss including surprisal, word length, and word frequency for reading time prediction.
"""

from typing import Dict, List

from transformers import PreTrainedModel, PreTrainedTokenizer
import pandas as pd
import torch
import numpy as np

from models.surprisal import compute_surprisal


def solve_beta(
    X: torch.Tensor, y: torch.Tensor, regularization: float = 1e-5
) -> torch.Tensor:
    """Solve for optimal coeffs beta^*"""
    # Compute X^T * X and add regularization
    XTX = torch.matmul(X.T, X) + regularization * torch.eye(
        X.shape[1], dtype=X.dtype, device=X.device
    )
    XTy = torch.matmul(X.T, y)
    L = torch.linalg.cholesky(XTX)
    return torch.cholesky_solve(XTy.unsqueeze(1), L).squeeze()


def compute_target_coeffs(
    surprisal: torch.Tensor,
    reading_times: torch.Tensor,
    word_lengths: torch.Tensor,
    word_frequencies: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute coefficients for target model"""
    # Construct the design matrix X_tgt
    ones = torch.ones_like(surprisal)
    X_tgt = torch.stack([surprisal, word_lengths, word_frequencies, ones], dim=1)

    # Compute coefficients
    beta_tgt = solve_beta(X_tgt, reading_times)
    beta_s, beta_l, beta_f, beta_0 = beta_tgt
    return beta_s, beta_l, beta_f, beta_0


def compute_baseline_coeffs(
    reading_times: torch.Tensor,
    word_lengths: torch.Tensor,
    word_frequencies: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute coefficients for baseline model"""
    ones = torch.ones_like(reading_times)
    X_base = torch.stack([word_lengths, word_frequencies, ones], dim=1)

    # Compute coefficients
    beta_base = solve_beta(X_base, reading_times)
    beta_l, beta_f, beta_0 = beta_base
    return beta_l, beta_f, beta_0


def compute_kl_full(target_lps: torch.Tensor, ref_lps: torch.Tensor) -> torch.Tensor:
    """Compute full KL divergence term"""
    # Terms are reversed in pytorch kl_div()
    return torch.nn.functional.kl_div(
        target_lps,
        ref_lps,
        reduction="none",
        log_target=True,
    ).sum(-1)


def compute_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch: Dict[str, np.ndarray],
    loss_type: str = "mse",
    reduction: str = "mean",
    mask_zero_reading_times: bool = False,
    mask_zero_freqs: bool = False,
    train_w_random_rts: bool = False,
    kl_weight: float = 0,
    kl_variant: str = "abs",
    model_ref: PreTrainedModel = None,
    is_eval: bool = False,
    mean_rt_train: float = None,
    std_rt_train: float = None,
) -> List[
    torch.Tensor,
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    pd.DataFrame,
    torch.Tensor,
]:
    """
    Compute loss for batch

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        batch (Dict[str, np.ndarray]): Current batch
        loss_type (str): Type of loss to compute
        reduction (str): Reduction type for loss
        mask_zero_reading_times (bool): Mask zero reading times
        mask_zero_freqs (bool): Mask zero frequencies
        train_w_random_rts (bool): Train with random reading times
        kl_weight (float): KL weight
        kl_variant (str): KL variant
        model_ref (PreTrainedModel): Reference model for KL divergence
        is_eval (bool): Is evaluation
        mean_rt_train (float): Mean reading time during training
        std_rt_train (float): Standard deviation of reading time during training

    Returns:
        List[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, pd.DataFrame, torch.Tensor]:
        Computed loss, coefficients, perplexity, cross entropy loss, sentence metrics, KL
    """
    reading_times = batch.pop("reading_times")

    if train_w_random_rts and not is_eval:
        assert (
            mean_rt_train is not None and std_rt_train is not None
        ), "Mean and std of training reading times required"
        random_reading_times = (
            torch.randn(
                reading_times.shape[0],
                reading_times.shape[1],
                dtype=reading_times.dtype,
            )
            * std_rt_train
            + mean_rt_train
        )
        random_reading_times[random_reading_times < 0] = 0
        random_reading_times[reading_times == -1] = -1
        reading_times = random_reading_times
        reading_times = reading_times.to(model.device)

    word_ids = batch.pop("word_ids")
    word_lengths = batch.pop("word_lengths")
    word_frequencies = batch.pop("log_unigram_frequencies")
    zero_frequencies = batch.pop("zero_freq_mask")

    surprisal, ce_loss, log_probs = compute_surprisal(
        batch,
        model,
        tokenizer,
        word_ids,
    )
    if kl_variant == "full":
        _, _, ref_logprobs = compute_surprisal(
            batch,
            model_ref,
            tokenizer,
            word_ids,
        )
    mask_word_ids = word_ids != -1
    # Apply mask to surprisal and token_word_mappings
    masked_surprisal = surprisal * mask_word_ids
    masked_word_ids = word_ids * mask_word_ids

    ppl_surprisal = torch.exp(torch.sum(masked_surprisal) / torch.sum(mask_word_ids))
    # Init summed_surprisal
    summed_surprisal = torch.zeros(
        surprisal.shape[0], surprisal.shape[1], device=surprisal.device
    )

    # Sum surprisal for tokens belonging to same word
    summed_surprisal.scatter_add_(1, masked_word_ids, masked_surprisal)

    mask_words = reading_times != -1
    if mask_zero_reading_times:
        mask_words = mask_words & (reading_times != 0)
    if mask_zero_freqs:
        mask_words = mask_words & (zero_frequencies != 0)

    masked_reading_times = reading_times[mask_words == 1]
    masked_lengths = word_lengths[mask_words == 1]
    masked_frequencies = word_frequencies[mask_words == 1]
    masked_surprisal = summed_surprisal[mask_words == 1]
    # Compute coefficients
    beta_s_trg, beta_l_trg, beta_f_trg, beta_0_trg = compute_target_coeffs(
        masked_surprisal,
        masked_reading_times,
        masked_lengths,
        masked_frequencies,
    )

    predictions_trg = (
        beta_s_trg * masked_surprisal
        + beta_l_trg * masked_lengths
        + beta_f_trg * masked_frequencies
        + beta_0_trg * torch.ones_like(masked_surprisal)
    )

    beta_l_bl, beta_f_bl, beta_0_bl = compute_baseline_coeffs(
        masked_reading_times, masked_lengths, masked_frequencies
    )

    coefficients_all = {
        "beta_s_trg": beta_s_trg.squeeze().detach().cpu(),
        "beta_l_trg": beta_l_trg.squeeze().detach().cpu(),
        "beta_f_trg": beta_f_trg.squeeze().detach().cpu(),
        "beta_0_trg": beta_0_trg.squeeze().detach().cpu(),
        "beta_l_bl": beta_l_bl.squeeze().detach().cpu(),
        "beta_f_bl": beta_f_bl.squeeze().detach().cpu(),
        "beta_0_bl": beta_0_bl.squeeze().detach().cpu(),
    }

    if loss_type == "mse":
        loss = torch.nn.functional.mse_loss(
            masked_reading_times, predictions_trg, reduction=reduction
        )
    else:
        raise ValueError("Invalid loss type")

    if kl_variant == "full":
        full_kl_mask = word_ids != -1
        full_kl_mask = torch.cat(
            [torch.tensor([[False]], device=model.device), full_kl_mask[:, :-1]],
            dim=1,
        )

        log_probs_sliced = log_probs[:, :-1, :]
        ref_logprobs_sliced = ref_logprobs[:, :-1, :]
        kl_full = compute_kl_full(
            log_probs_sliced[full_kl_mask].unsqueeze(0),
            ref_logprobs_sliced[full_kl_mask].unsqueeze(0),
        )  # Numerical precision can lead to very small negative values
        kl = kl_full.mean()
    else:
        kl = torch.tensor(0)
    if not is_eval and kl_weight > 0:
        loss = loss + kl_weight * kl

    # Fit regression models, remove additional dimensions, assume batch size is 1 for now
    unpadded_surprisal = masked_surprisal.squeeze(0).detach().cpu().numpy()
    unpadded_reading_times = masked_reading_times.squeeze(0).detach().cpu().numpy()
    unpadded_word_lengths = masked_lengths.squeeze(0).cpu().numpy()
    unpadded_word_frequencies = masked_frequencies.squeeze(0).cpu().numpy()

    sentence_metrics = pd.DataFrame(
        {
            "surprisal": unpadded_surprisal,
            "reading_times": unpadded_reading_times,
            "word_length": unpadded_word_lengths,
            "word_frequency": unpadded_word_frequencies,
        }
    )
    return loss, coefficients_all, ppl_surprisal, ce_loss, sentence_metrics, kl
