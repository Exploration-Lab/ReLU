import os
import json
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader

from .mcqa import mcqa
from .cloze import cloze
from .analogy import analogy
from .odd_one_out import oddoneout
from .comprehension import comprehension


class Qualitative(Dataset):
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
        qa_format = item["qa"]
        mcqa_format = item["mcq"]
        cloze_format = item["cloze"]
        # analogy_format = item["analogy"]
        # odd_one_out_format = item["odd"]
        # comprehension_format = item["comprehension"]
        return (
            self.format_prompt_qa(qa_format),
            self.format_prompt_mcqa(mcqa_format),
            self.format_prompt_cloze(cloze_format),
        )

    def load_dataset(self, dataset_path):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        return dataset

    def format_prompt_mcqa(self, mcqa_format):
        question = mcqa_format["question"]
        try:
            options = [
                mcqa_format["A"],
                mcqa_format["B"],
                mcqa_format["C"],
                mcqa_format["D"],
            ]
        except:
            raise ValueError(
                f"Options not found for question {question} in dataset {self.dataset_path}"
            )
        correct_option = mcqa_format["Correct option"][0]

        answer = mcqa_format["answer"]
        answer = mcqa_format[mcqa_format["Correct option"][0]]

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

    def format_prompt_cloze(self, cloze_format):
        question = cloze_format["question"]
        answer = cloze_format["blanks"][0]
        mask = cloze_format["mask"]
        try:
            prompt_template_text = (
                f"Question: {question}\nAnswer: {mask[:mask.index('[MASK]')]}"
            )
        except ValueError:
            breakpoint()
            raise ValueError(
                f"Answer {answer} not found in mask {mask} for question {question} in dataset {self.dataset_path}"
            )

        return prompt_template_text, answer

    def format_prompt_qa(self, qa_format):
        question = qa_format["question"]
        answer = qa_format["answer"]
        formatted_prompt = f"Question: {question}\nAnswer:"
        return formatted_prompt, answer


class GenerationTrackerQualitative:
    """
    Class to keep track of the generated text qualitative analysis
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.generated_text = []

    def update(
        self,
        qa_prompt,
        qa_label,
        mcqa_prompt,
        mcqa_label,
        cloze_prompt,
        cloze_label,
        qa_generated_text,
        mcqa_generated_text,
        cloze_generated_text,
    ):
        self.generated_text.append(
            {
                "qa_prompt": qa_prompt,
                "qa_label": qa_label,
                "mcqa_prompt": mcqa_prompt,
                "mcqa_label": mcqa_label,
                "cloze_prompt": cloze_prompt,
                "cloze_label": cloze_label,
                "qa_generated_text": qa_generated_text,
                "mcqa_generated_text": mcqa_generated_text,
                "cloze_generated_text": cloze_generated_text,
            }
        )

    def save(self):
        print(f"Saving the generated text to {self.save_path}")
        with open(self.save_path, "w") as f:
            json.dump(self.generated_text, f, indent=4)
