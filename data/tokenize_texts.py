from typing import List, Dict, Any
import math

import pandas as pd
import numpy as np
from wordfreq import word_frequency
from transformers import PreTrainedTokenizer


def tokenize_texts(
    texts: List[List[str]],
    reading_times: List[List[float]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> Dict[str, np.ndarray]:
    """
    Tokenizes texts and returns a dictionary with the tokenized texts, attention masks,
    reading times, token to word mappings, word lengths, and log unigram frequencies.
    Args:
        texts (list): List of lists of words.
        reading_times (list): List of lists of reading times.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
        max_length (int): Maximum length of the tokenized texts.
    Returns:
        dict: Dictionary with the tokenized texts, attention masks,
        reading times, token to word mappings, word lengths, and log unigram frequencies.
    """
    tokenized_texts = []
    token_word_ids = []
    attention_masks = []
    word_lengths = []
    log_unigram_frequencies = []
    all_reading_times = []
    all_words = []
    all_zero_freqs = []

    for sent_index, words in enumerate(texts):
        # Skip short sents
        if len(words) <= 3:
            continue
        elif len(words) > max_length - 1:
            print("Skipping sentence with more than max_length words")
            continue
        # exclude if contains nan
        elif any([pd.isna(word) for word in words]):
            continue

        joined_sentence = " ".join(words)
        tokens = tokenizer(
            joined_sentence,
            padding="max_length",
            truncation=False,  # Check for length later
            max_length=max_length,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        ids = np.array(tokens["input_ids"])
        unpadded_ids = ids[ids != tokenizer.pad_token_id]
        if (
            len(unpadded_ids) > max_length
        ):  # exlude sentences with more than max_length tokens
            print("Skipping sentence with more than max_length tokens")
            continue
        word_ids = np.array(tokens.word_ids())
        offset_mapping = tokens["offset_mapping"]
        offset_words = [joined_sentence[start:end] for start, end in offset_mapping]
        corrected_word_ids = []
        corrected_words = []
        words_index = 0
        offset_index = 0
        try:
            while words_index < len(words):
                word = words[words_index]
                if words_index > 0:
                    word = " " + word
                if word == offset_words[offset_index]:
                    corrected_word_ids.append(words_index)
                    corrected_words.append(word.lstrip())
                    words_index += 1
                    offset_index += 1
                else:
                    target_word = word
                    merged_word = offset_words[offset_index]
                    merged_word_id = [words_index]
                    offset_index += 1
                    while target_word != merged_word:
                        merged_word += offset_words[offset_index]
                        merged_word_id.append(words_index)
                        offset_index += 1
                    corrected_words.append(merged_word.lstrip())
                    corrected_word_ids.extend(merged_word_id)
                    words_index += 1
            assert (
                corrected_words == words
            ), "Corrected words do not match the original words."
            assert max(corrected_word_ids) + 1 == len(
                words
            ), "Highest index in corrected word IDs does not match the number of words."
            assert len(corrected_word_ids) == len(
                word_ids[word_ids != None]
            ), "Length mismatch between corrected word IDs and word IDs."
            assert (
                len(reading_times[sent_index]) == max(corrected_word_ids) + 1
            ), "Reading times length does not match the number of words."
            ids_checker = {}
            for index, word_id in enumerate(corrected_word_ids):
                if word_id not in ids_checker:
                    ids_checker[word_id] = []
                ids_checker[word_id].append(tokens["input_ids"][index])
            for word_id, word_ids in ids_checker.items():
                if words[word_id].strip() != tokenizer.decode(word_ids).strip():
                    print("Mismatch between word and tokenized word.")
                    print("Word: ", words[word_id].strip())
                    print("Tokenized word: ", tokenizer.decode(word_ids).strip())
                    raise ValueError("Mismatch between word and tokenized word.")

        except Exception as e:
            print("Encountered an error while correcting word IDs.: ", e)
            print("Skipping sentence: ", words)
            continue

        corrected_word_ids = np.pad(
            corrected_word_ids,
            (0, max_length - len(corrected_word_ids)),
            "constant",
            constant_values=-1,
        )

        # Process word lengths and frequencies
        lengths = [len(word) for word in words]
        freqs = [word_frequency(word, "en", wordlist="best") for word in words]
        zero_freqs = np.where(np.array(freqs) == 0, 0, 1)
        all_zero_freqs.append(
            np.pad(
                zero_freqs,
                (0, max_length - len(zero_freqs)),
                "constant",
                constant_values=0,
            )
        )
        log_freqs = [
            -math.log2(frequency if frequency > 0 else 1e-10) for frequency in freqs
        ]
        token_word_ids.append(corrected_word_ids)
        tokenized_texts.append(tokens["input_ids"])
        attention_masks.append(tokens["attention_mask"])
        word_lengths.append(
            np.pad(
                lengths, (0, max_length - len(lengths)), "constant", constant_values=-1
            )
        )
        log_unigram_frequencies.append(
            np.pad(
                log_freqs,
                (0, max_length - len(log_freqs)),
                "constant",
                constant_values=-1,
            )
        )

        # Process reading times, pad and assert length matches number of words
        current_reading_times = reading_times[sent_index]
        assert (
            len(current_reading_times) == max(corrected_word_ids) + 1
        ), "Reading times length does not match the number of words."

        all_reading_times.append(
            np.pad(
                current_reading_times,
                (0, max_length - len(current_reading_times)),
                "constant",
                constant_values=-1,
            )
        )

        # Save words to for logging data splits
        all_words.append(
            np.pad(words, (0, max_length - len(words)), "constant", constant_values=-1)
        )

    return {
        "input_ids": np.asarray(tokenized_texts, dtype=np.int64),
        "attention_mask": np.asarray(attention_masks, dtype=np.int64),
        "reading_times": np.asarray(all_reading_times, dtype=np.float64),
        "word_ids": np.asarray(token_word_ids, dtype=np.int64),
        "word_lengths": np.asarray(word_lengths, dtype=np.int64),
        "log_unigram_frequencies": np.asarray(
            log_unigram_frequencies, dtype=np.float64
        ),
        "zero_freq_mask": np.asarray(all_zero_freqs, dtype=np.int64),
        "words": np.asarray(all_words, dtype=object),
    }
