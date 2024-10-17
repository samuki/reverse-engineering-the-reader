from typing import Dict, Tuple
import os

import torch
from torch.utils.data import DataLoader, random_split
from transformers import PreTrainedTokenizer

from data.load_provo import process_provo_corpus, ProvoDataset
from data.load_dundee import process_dundee_corpus, DundeeDataset
from data.load_zuco import process_zuco_corpus, ZucoDataset
from data.data_utils import (
    save_data_split_logs,
    log_dataloader,
    get_mean_std_reading_times_dataloader,
)


def process_data(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    dataset: str,
    evaluate: bool = False,
):
    if dataset == "provo":
        data = process_provo_corpus(
            tokenizer,
            config,
            evaluate=evaluate,
        )
        dataset = ProvoDataset(data)
    elif dataset == "dundee":
        data = process_dundee_corpus(
            tokenizer,
            config,
            evaluate=evaluate,
        )
        dataset = DundeeDataset(data)
    elif dataset == "zuco":
        data = process_zuco_corpus(
            tokenizer,
            config,
            evaluate=evaluate,
        )
        dataset = ZucoDataset(data)
    else:
        raise ValueError("Invalid dataset specified in config file.")
    return dataset


def load_data(
    config: Dict, tokenizer: PreTrainedTokenizer
) -> Tuple[DataLoader, DataLoader]:
    experiment_name = config["experiment_name"]
    seed = config["seed"]
    preference_dataset_name = config["training_kwargs"].get("preference_dataset", False)
    eval_dataset_name = config["training_kwargs"]["eval_dataset"]

    dataset = process_data(config, tokenizer, preference_dataset_name)

    eval_data = process_data(config, tokenizer, eval_dataset_name, evaluate=True)

    # We split the train dataset again.
    eval_size = int(config["training_kwargs"]["eval_split_size"] * len(dataset))
    train_size = len(dataset) - eval_size
    generator = torch.Generator().manual_seed(seed)  # Shuffle according to seed
    train_dataset, _ = random_split(
        dataset, [train_size, eval_size], generator=generator
    )
    eval_generator = torch.Generator().manual_seed(12321)  # Same ordering across runs
    eval_dataset, _ = random_split(
        eval_data, [len(eval_data), 0], generator=eval_generator
    )

    # Log datasplits
    log_data_splits = config["training_kwargs"].get("log_data_splits", True)
    if log_data_splits:
        os.makedirs("data_split_logs", exist_ok=True)
        os.makedirs(
            f"data_split_logs/{experiment_name}_{seed}",
            exist_ok=True,
        )
        save_data_split_logs(
            train_dataset,
            eval_dataset,
            experiment_name,
            eval_dataset_name,
            seed,
        )

    train_batch_size = config["training_kwargs"].get("train_batch_size", 1)
    assert train_batch_size == 1, "Only batch size 1 is supported for now."

    eval_batch_size = config["training_kwargs"].get("eval_batch_size", 1)
    assert eval_batch_size == 1, "Only batch size 1 is supported for now."

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    # Log dataloaders to inspect
    log_data_loader = config["training_kwargs"].get("log_data_loader", True)
    mean_reading_time, std_reading_time = get_mean_std_reading_times_dataloader(
        train_loader
    )
    if log_data_loader:
        os.makedirs("data_split_logs", exist_ok=True)
        os.makedirs(
            f"data_split_logs/{experiment_name}_{seed}",
            exist_ok=True,
        )
        log_dataloader(
            train_loader,
            "train_loader_log.json",
            experiment_name,
            tokenizer,
            seed,
        )
        log_dataloader(
            eval_loader,
            "eval_loader_log.json",
            experiment_name,
            tokenizer,
            seed,
        )

    return train_loader, eval_loader, mean_reading_time, std_reading_time
