import os
import json
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader


class cloze(Dataset):
    def __init__(self, dataset_path, seed=42):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset(dataset_path)
        self.seed = seed
        np.random.seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer = item["blanks"][0]
        mask = item["mask"]
        prompt = self.format_prompt(question, mask, answer)
        return prompt, answer

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        # remove the entries from the dataset where the mask does not contain the [MASK]
        dataset = [
            item
            for item in dataset
            if "[MASK]" in item["mask"] and item["mask"].count("[MASK]") == 1
        ]
        return dataset

    def format_prompt(self, question, mask, answer):
        try:
            prompt_template_text = (
                f"Question: {question}\nAnswer: {mask[:mask.index('[MASK]')]}"
            )
        except ValueError:
            breakpoint()
            raise ValueError(
                f"Answer {answer} not found in mask {mask} for question {question} in dataset {self.dataset_path}"
            )
        return prompt_template_text


class PerformanceTrackerCloze(object):
    """
    Class to keep track of the performance of a model on cloze task
    the sequence’s probability is normalized for length by taking the nth root, or P(x1,x2,...,xn) = pn Qni=1 P(xi).
    """

    def __init__(self):
        self.normalized_sequence_probability = 0.0
        self.num_sequences = 0

    def update(self, sequence_probability_list):
        self.num_sequences += 1
        sequence_length = len(sequence_probability_list)
        # sequence_probability = torch.prod(torch.tensor(sequence_probability_list))
        sequence_probability = (
            torch.log(torch.tensor(sequence_probability_list)).sum().exp()
        )
        self.normalized_sequence_probability += sequence_probability ** (
            1 / sequence_length
        )

    def get_performance(self):
        return self.normalized_sequence_probability / self.num_sequences
