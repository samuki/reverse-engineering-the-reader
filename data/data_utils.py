from typing import Any
import json

import numpy as np
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


def save_data_split_logs(
    train_dataset: Any,
    eval_dataset: Any,
    experiment_name: str,
    eval_dataset_name: str,
    seed: int,
):
    sents_train = train_dataset.dataset.data.pop("words")
    indices_train = train_dataset.indices

    sents_eval = eval_dataset.dataset.data.pop("words")
    indices_eval = eval_dataset.indices
    all_train_sents = []
    all_eval_sents = []

    all_train_reading_times = []
    rts_train = train_dataset.dataset.data["reading_times"]
    zfs_train = train_dataset.dataset.data["zero_freq_mask"]

    all_eval_reading_times = []

    rts_eval = eval_dataset.dataset.data["reading_times"]
    zfs_eval = eval_dataset.dataset.data["zero_freq_mask"]

    for train_index in indices_train:
        sent = sents_train[train_index]
        rts_index = rts_train[train_index]
        zfs_index = zfs_train[train_index]
        for i, rt_train in enumerate(rts_index):
            if rt_train > 0 and zfs_index[i] > 0:
                all_train_reading_times.append(rt_train)
        all_train_sents.append([word for word in sent.tolist() if word != "-1"])
    for eval_index in indices_eval:
        sent = sents_eval[eval_index]
        rts_index_eval = rts_eval[eval_index]
        zfs_index_eval = zfs_eval[eval_index]

        for j, rt_eval in enumerate(rts_index_eval):
            if rt_eval > 0 and zfs_index_eval[j] > 0:
                all_eval_reading_times.append(rt_eval)
        all_eval_sents.append([word for word in sent.tolist() if word != "-1"])

    with open(
        f"data_split_logs/{experiment_name}_{seed}/train_words.txt",
        "w",
    ) as f:
        f.write("\n".join([" ".join(train_sent) for train_sent in all_train_sents]))
    with open(
        f"data_split_logs/{experiment_name}_{seed}/eval_words.txt",
        "w",
    ) as f:
        f.write("\n".join([" ".join(eval_sent) for eval_sent in all_eval_sents]))
    data_stats = {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "train_words": sum([len(sent) for sent in all_train_sents]),
        "eval_words": sum([len(sent) for sent in all_eval_sents]),
        "train_words_included": len(all_train_reading_times),
        "eval_words_included": len(all_eval_reading_times),
    }
    with open(
        f"data_split_logs/{experiment_name}_{seed}/data_stats.json",
        "w",
    ) as f:
        json.dump(data_stats, f)


def get_mean_std_reading_times_dataloader(loader: DataLoader):
    reading_times = []
    for batch in loader:
        reading_times.extend(
            batch["reading_times"][batch["reading_times"] > 0].tolist()
        )
    return np.mean(reading_times), np.std(reading_times)


def log_dataloader(
    loader: DataLoader,
    filename: str,
    experiment_name: str,
    tokenizer: PreTrainedTokenizer,
    seed: int,
):
    log_file_path = f"data_split_logs/{experiment_name}_{seed}/{filename}"
    with open(log_file_path, "w") as log_file:
        for batch_idx, batch in enumerate(loader):
            batch_data = {
                "batch_idx": batch_idx,
                "decoded": tokenizer.batch_decode(
                    batch["input_ids"], skip_special_tokens=True
                ),
            }
            log_file.write(json.dumps(batch_data) + "\n")
