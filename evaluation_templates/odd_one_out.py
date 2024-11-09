import os
import json
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader


class oddoneout(Dataset):
    def __init__(self, dataset_path, shuffle_options=True, n_options=4, seed=42):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset(dataset_path)
        self.shuffle_options = shuffle_options
        self.n_options = n_options
        self.seed = seed
        np.random.seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = item["ans"]
        options = [
            item["A"]["answer"],
            item["B"]["answer"],
            item["C"]["answer"],
            item["D"]["answer"],
        ]
        prompt, label = self.format_prompt(options, label)
        return prompt, label

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def format_prompt(self, options, label):
        correct_option = ord(label) - 65
        answer = options[correct_option]
        wrong_options = [i for i in range(self.n_options) if i != correct_option]

        # sample wrong options based on n_options
        wrong_options = np.random.choice(
            wrong_options, self.n_options - 1, replace=False
        )
        options = [options[correct_option]] + [
            options[wrong_option] for wrong_option in wrong_options
        ]

        if self.shuffle_options:
            np.random.shuffle(options)

        correct_option = options.index(answer)

        # prompt_template_text = f"Odd one out EXAM:\n"
        # prompt_template_text = (
        #     "Question: Choose the option that is providing different information from the rest of the options (choose the odd option from the below options):"
        #     + "\n"
        # )
        prompt_template_text = (
            "Question: Choose the option that provides a correct fact (choose the correct option from the below options):"
            + "\n"
        )

        for i, option in enumerate(options):
            # A. option1 \n B. option2 \n C. option3 \n D. option4 \n
            prompt_template_text += f"{chr(65 + i)}. {option}\n"
        prompt_template_text += "Answer:"

        label = chr(65 + correct_option)
        return prompt_template_text, label


class PerformanceTrackerOddOneOut(object):
    """
    Class to keep track of the performance of a model on multiple choice question answering tasks
    the probability over the options is used to determine the predicted label
    """

    def __init__(
        self,
    ):
        self.total_per_class = {}
        self.correct_per_class = {}
        self.total = 0
        self.correct = 0
        self.accuracy = 0.0

    def update(self, label, predicted_label):
        if label not in self.total_per_class:
            self.total_per_class[label] = 0
            self.correct_per_class[label] = 0
        self.total_per_class[label] += 1
        self.total += 1
        if label == predicted_label:
            self.correct_per_class[label] += 1
            self.correct += 1

    def get_performance(self):
        self.accuracy = self.correct / self.total
        return self.accuracy

    def get_performance_per_class(self):
        accuracy_per_class = {}
        for label in self.total_per_class:
            accuracy_per_class[label] = (
                self.correct_per_class[label] / self.total_per_class[label]
            )
        return accuracy_per_class

    def reset(self):
        self.total_per_class = {}
        self.correct_per_class = {}
        self.total = 0
        self.correct = 0
        self.accuracy = 0.0
        return self
