from typing import Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer
import torch


def compute_surprisal(
    inputs: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    word_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute surprisal values given the inputs and model.

    Args:
        inputs (dict): Tokenized inputs.
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        word_ids (torch.Tensor): Word ids for the input, only used as mask for loss

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of token surprisal, loss, and log probabilities

    """

    # Manually add bos token to input since add_special tokens has no effect for this tokenizer
    input_ids = torch.cat(
        (
            torch.tensor([[tokenizer.bos_token_id]], device=model.device),
            inputs["input_ids"],
        ),
        dim=1,
    )
    bos_tokens_tensor = torch.tensor(
        [[tokenizer.bos_token_id]] * inputs["input_ids"].size(dim=0)
    ).to(model.device)
    # Mask bos token
    attention_mask = torch.cat(
        [
            torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(model.device),
            inputs["attention_mask"],
        ],
        dim=1,
    )

    # Mask labels with -100 where input ids are -1 for correct ce loss and perplexity calculation
    mask = word_ids != -1
    mask = torch.cat(
        [
            torch.ones(bos_tokens_tensor.size(), dtype=torch.bool).to(model.device),
            mask,
        ],
        dim=1,
    )
    labels = torch.where(mask, input_ids, -100)

    # Compute logits and loss
    outputs = model(
        **{"input_ids": input_ids, "attention_mask": attention_mask}, labels=labels
    )

    # Shift tokens
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    log_probs_shifted = log_probs[:, :-1, :]
    input_ids_shifted = input_ids[:, 1:]

    # Gather log probabilities for the correct tokens
    surprisal = torch.mul(
        -1,
        torch.gather(log_probs_shifted, 2, input_ids_shifted[:, :, None]).squeeze(-1),
    )
    return surprisal, outputs.loss, log_probs
