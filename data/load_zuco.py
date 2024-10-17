from typing import Dict

from torch.utils.data import Dataset
import pandas as pd
from transformers import PreTrainedTokenizer
import numpy as np

from data.tokenize_texts import tokenize_texts


class ZucoDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data

    def __len__(self):
        return self.data["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.data["input_ids"][idx, :],
            "attention_mask": self.data["attention_mask"][idx, :],
            "reading_times": self.data["reading_times"][idx, :],
            "word_ids": self.data["word_ids"][idx, :],
            "word_lengths": self.data["word_lengths"][idx, :],
            "log_unigram_frequencies": self.data["log_unigram_frequencies"][idx, :],
            "zero_freq_mask": self.data["zero_freq_mask"][idx, :],
        }


def process_zuco_corpus(
    tokenizer: PreTrainedTokenizer, config: Dict, evaluate: bool = False
) -> Dict[str, np.ndarray]:
    if evaluate:
        file_path = config["training_kwargs"]["eval_data_path"]
        target = config["training_kwargs"]["train_dataset_target"]
    else:
        file_path = config["training_kwargs"]["data_path"]
        target = config["training_kwargs"]["eval_dataset_target"]

    max_length = config["training_kwargs"].get("max_length", 512)
    zuco_data = pd.read_csv(file_path)

    texts = zuco_data.groupby("sentence_num")["word"].apply(list).tolist()
    reading_times = zuco_data.groupby("sentence_num")[target].apply(list).tolist()

    return tokenize_texts(
        texts,
        reading_times,
        tokenizer,
        max_length=max_length,
    )
