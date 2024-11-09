import os
import json
import torch

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from torch.utils.data import Dataset, DataLoader


class comprehension(Dataset):
    def __init__(
        self, dataset_path, format="mcqa", n_options=4, shuffle_options=True, seed=42
    ):
        self.dataset_path = dataset_path
        self.dataset = self.load_dataset(dataset_path)
        self.n_options = n_options
        self.shuffle_options = shuffle_options
        self.dataset_qa, self.dataset_mcq = self.convert_dataset_to_prompt_format(
            self.dataset
        )
        if format == "mcqa":
            self.dataset = self.dataset_mcq
        elif format == "qa":
            self.dataset = self.dataset_qa
        else:
            raise ValueError(f"Invalid format {format}")
        self.seed = seed
        np.random.seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def convert_dataset_to_prompt_format(self, dataset):
        """
        every question is considered as a single prompt, the output style is as follows:
        Context: <prompt>
        Question: <question>
        Answer: <answer>
        """
        formatted_dataset_qa = []
        for item in dataset:
            prompt = item["prompt"]
            for qa_pair in item["QA"]:
                question = qa_pair["question"]
                answer = qa_pair["answer"]
                formatted_prompt = f"Context: {prompt}\nQuestion: {question}\nAnswer:"
                label = answer
                formatted_dataset_qa.append((formatted_prompt, label))
        formatted_dataset_mcq = []
        for item in dataset:
            prompt = item["prompt"]
            for mcq_question in item["mcq"]:
                question = mcq_question["question"]
                try:
                    options = [
                        mcq_question["A"],
                        mcq_question["B"],
                        mcq_question["C"],
                        mcq_question["D"],
                    ]
                except:
                    raise ValueError(
                        f"Options not found for question {question} in dataset {self.dataset_path}"
                    )

                correct_option = mcq_question["Correct option"][0]
                answer = mcq_question[mcq_question["Correct option"][0]]
                formatted_prompt, label = self.format_prompt_mcqa(
                    question, options, answer, correct_option
                )
                formatted_prompt = f"Context: {prompt}\n{formatted_prompt}"
                formatted_dataset_mcq.append((formatted_prompt, label))

        return formatted_dataset_qa, formatted_dataset_mcq

    def format_prompt_mcqa(self, question, options, answer, correct_option):
        try:
            correct_option = options.index(answer)
        except ValueError:
            raise ValueError(
                f"Answer {answer} not found in options {options} for question {question} in dataset {self.dataset_path}"
            )
        # correct_option = ord(correct_option) - 65

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

        prompt_template_text = f"Question: {question} \n"
        for i, option in enumerate(options):
            # A. option1 \n B. option2 \n C. option3 \n D. option4 \n
            prompt_template_text += f"{chr(65 + i)}. {option}\n"
        prompt_template_text += "Answer:"

        label = chr(65 + correct_option)

        return prompt_template_text, label


class PerformanceTrackerComprehension(object):
    """
    Class to keep track of the performance of a model on comprehension task
    the comprehension has two set of questions, one is QA and other is MCQ
    QA is a list of questions and answers evaluated using ROUGE
    MCQ is a list of multiple choice questions evaluated using accuracy
    """

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.rouge_scores = []
        self.total_per_class = {}
        self.correct_per_class = {}
        self.total = 0
        self.correct = 0
        self.accuracy = 0.0

        self.predictions_df = pd.DataFrame(columns=["prompt", "prediction", "target"])

    def update_qa(self, prompt, predictions, targets):
        rouge_score = self.calculate_rouge_score(predictions, targets)
        self.rouge_scores.append(rouge_score)
        self.add_predictions(prompt, predictions, targets)

    def update_mcq(self, predictions, targets):
        self.total += 1
        if predictions == targets:
            self.correct += 1
        if targets not in self.total_per_class:
            self.total_per_class[targets] = 0
            self.correct_per_class[targets] = 0
        self.total_per_class[targets] += 1
        if predictions == targets:
            self.correct_per_class[targets] += 1
        self.accuracy = self.correct / self.total

    def calculate_rouge_score(self, predictions, targets):
        scores = self.rouge_scorer.score(predictions, targets)
        rouge_score = (
            scores["rouge1"].fmeasure
            + scores["rouge2"].fmeasure
            + scores["rougeL"].fmeasure
        ) / 3
        return rouge_score

    def get_mcqa_performance(self):
        return self.accuracy

    def get_mcqa_performance_per_class(self):
        accuracy_per_class = {}
        for label in self.total_per_class:
            accuracy_per_class[label] = (
                self.correct_per_class[label] / self.total_per_class[label]
            )
        return accuracy_per_class

    def get_qa_performance(self):
        return np.mean(self.rouge_scores)

    def add_predictions(self, prompt, predictions, targets):
        # save predictions to a csv file for qa format, prompt, prediction, target
        # create a data frame and save it to a csv file
        self.predictions_df = self.predictions_df._append(
            {
                "prompt": prompt.replace("\n", "\\n"),
                "prediction": predictions.replace("\n", "\\n"),
                "target": targets.replace("\n", "\\n"),
            },
            ignore_index=True,
        )

    def save_predictions(self, save_file_path):
        self.predictions_df.to_csv(save_file_path, index=False)
        print(f"Predictions saved to {save_file_path}")
