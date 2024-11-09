import argparse
import sys
import os
import torch
import random
import numpy as np

from tqdm import tqdm

from utils import load_model_and_tokenizer
from evaluation_templates.mcqa import mcqa, PerformanceTrackerMCQA
from evaluation_templates.cloze import cloze, PerformanceTrackerCloze
from evaluation_templates.comprehension import (
    comprehension,
    PerformanceTrackerComprehension,
)
from evaluation_templates.analogy import analogy, PerformanceTrackerAnalogy
from evaluation_templates.odd_one_out import oddoneout, PerformanceTrackerOddOneOut
from evaluation_templates.qualitative import Qualitative, GenerationTrackerQualitative


def evaluate_prompt(model, tokenizer, args, device, verbose=True):

    if args.reformat_type == "mcqa":
        dataset = mcqa(args.dataset_path, n_options=args.num_options)
        performance_tracker = PerformanceTrackerMCQA()
    elif args.reformat_type == "cloze":
        dataset = cloze(args.dataset_path)
        performance_tracker = PerformanceTrackerCloze()
    elif args.reformat_type == "odd-one-out":
        dataset = oddoneout(args.dataset_path, n_options=args.num_options)
        performance_tracker = PerformanceTrackerOddOneOut()
    elif args.reformat_type == "analogy-mcqa":
        dataset = analogy(args.dataset_path, n_options=args.num_options)
        performance_tracker = PerformanceTrackerAnalogy()
    elif args.reformat_type == "comprehension-mcqa":
        dataset = comprehension(
            args.dataset_path, format="mcqa", n_options=args.num_options
        )
        performance_tracker = PerformanceTrackerComprehension()
        performance_tracker.get_performance = performance_tracker.get_mcqa_performance
    elif args.reformat_type == "comprehension-qa":
        dataset = comprehension(args.dataset_path, format="qa")
        performance_tracker = PerformanceTrackerComprehension()
        performance_tracker.get_performance = performance_tracker.get_qa_performance
    else:
        print("Invalid reformat_type")
        sys.exit()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    model.eval()
    correct = 0
    total = 0
    for i, (prompt, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        prompt = prompt[0]
        label = label[0]

        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        tokenized_prompt = {k: v.to(device) for k, v in tokenized_prompt.items()}
        if args.reformat_type == "mcqa":
            with torch.no_grad():
                outputs = model(**tokenized_prompt)
                generated_logits = outputs.logits

            option_ids = [tokenizer.encode(f" {chr(65 + i)}")[-1] for i in range(4)]
            option_logits = generated_logits[:, -1, option_ids]
            generated_choice = f"{chr(65 + torch.argmax(option_logits).item())}"
            performance_tracker.update(generated_choice, label)
        elif args.reformat_type == "cloze":
            # tokenize the labels
            tokenized_label = tokenizer(label, return_tensors="pt", padding=True)
            for key, value in tokenized_label.items():
                tokenized_label[key] = value.to(device)

            with torch.no_grad():
                generated_logits_sequence = []
                for idx in range(tokenized_label["input_ids"].shape[-1]):
                    outputs = model(**tokenized_prompt)
                    tokenized_prompt["input_ids"] = torch.cat(
                        [
                            tokenized_prompt["input_ids"],
                            tokenized_label["input_ids"][:, idx].unsqueeze(0),
                        ],
                        dim=1,
                    )
                    generated_logits_sequence.append(outputs.logits)
            generated_logits_sequence = torch.cat(generated_logits_sequence, dim=1)
            generated_probabilities = torch.nn.functional.softmax(
                generated_logits_sequence, dim=-1
            )
            # get the probability of the correct sequence
            correct_sequence_probability = []
            for idx in range(tokenized_label["input_ids"].shape[-1]):
                correct_sequence_probability.append(
                    generated_probabilities[
                        0,
                        -len(tokenized_label["input_ids"][0]) + idx,
                        tokenized_label["input_ids"][0, idx],
                    ]
                )

            performance_tracker.update(correct_sequence_probability)
        elif args.reformat_type == "comprehension-mcqa":
            with torch.no_grad():
                outputs = model(**tokenized_prompt)
                generated_logits = outputs.logits

            option_ids = [tokenizer.encode(f" {chr(65 + i)}")[-1] for i in range(4)]
            option_logits = generated_logits[:, -1, option_ids]
            generated_choice = f"{chr(65 + torch.argmax(option_logits).item())}"
            performance_tracker.update_mcq(generated_choice, label)
        elif args.reformat_type == "comprehension-qa":
            tokenized_label = tokenizer(label, return_tensors="pt", padding=True)
            with torch.no_grad():
                predictions = model.generate(
                    **tokenized_prompt,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=tokenized_label.input_ids.shape[-1],
                )
            generated_answer_text = tokenizer.decode(
                predictions[0][len(tokenized_prompt["input_ids"][0]) :]
            )

            performance_tracker.update_qa(prompt, generated_answer_text, label)
        elif args.reformat_type == "analogy-mcqa":
            with torch.no_grad():
                outputs = model(**tokenized_prompt)
                generated_logits = outputs.logits

            option_ids = [tokenizer.encode(f" {chr(65 + i)}")[-1] for i in range(4)]
            option_logits = generated_logits[:, -1, option_ids]
            generated_choice = f"{chr(65 + torch.argmax(option_logits).item())}"
            performance_tracker.update(generated_choice, label)
        elif args.reformat_type == "odd-one-out":
            with torch.no_grad():
                outputs = model(**tokenized_prompt)
                generated_logits = outputs.logits

            option_ids = [tokenizer.encode(f" {chr(65 + i)}")[-1] for i in range(4)]
            option_logits = generated_logits[:, -1, option_ids]
            generated_choice = f"{chr(65 + torch.argmax(option_logits).item())}"
            performance_tracker.update(generated_choice, label)

    if verbose:
        if args.reformat_type == "mcqa":
            print(
                f"Success Rate for {args.dataset_path}: {performance_tracker.get_performance()}"
            )
            # get accuracy per class
            performance_per_class = performance_tracker.get_performance_per_class()
            for key, value in performance_per_class.items():
                print(f"Accuracy for class {key}: {value}")
        elif args.reformat_type == "cloze":
            print(
                f"Sequence Prob. for {args.dataset_path}: {performance_tracker.get_performance()}"
            )
        elif args.reformat_type == "comprehension-mcqa":
            print(
                f"Success Rate for {args.dataset_path}: {performance_tracker.get_mcqa_performance()}"
            )
            # get accuracy per class
            performance_per_class = performance_tracker.get_mcqa_performance_per_class()
            for key, value in performance_per_class.items():
                print(f"Accuracy for class {key}: {value}")
        elif args.reformat_type == "comprehension-qa":
            print(
                f"ROUGE Score for {args.dataset_path}: {performance_tracker.get_qa_performance()}"
            )
        elif args.reformat_type == "analogy-mcqa":
            print(
                f"Success Rate for {args.dataset_path}: {performance_tracker.get_performance()}"
            )
            # get accuracy per class
            performance_per_class = performance_tracker.get_performance_per_class()
            for key, value in performance_per_class.items():
                print(f"Accuracy for class {key}: {value}")
        elif args.reformat_type == "odd-one-out":
            print(
                f"Success Rate for {args.dataset_path}: {performance_tracker.get_performance()}"
            )
            # get accuracy per class
            performance_per_class = performance_tracker.get_performance_per_class()
            for key, value in performance_per_class.items():
                print(f"Accuracy for class {key}: {value}")
    if args.reformat_type == "comprehension-qa":
        performance_tracker.save_predictions(
            save_file_path=f"./logs/{os.path.basename(args.dataset_path)[:-4]}---{str('--'.join(args.model_path.split('/')[-2:]))}.csv"
        )
    return performance_tracker.get_performance()


def qualitative_analysis(model, tokenizer, args, device):
    qualitative = Qualitative(args.dataset_path)
    save_file_path = f"./logs/qualitative_analysis/qualitative_analysis_results--{os.path.basename(args.dataset_path)[:-4]}---{str('--'.join(args.model_path.split('/')[-2:]))}.json"
    generation_tracker = GenerationTrackerQualitative(save_file_path)
    dataloader = torch.utils.data.DataLoader(
        qualitative, batch_size=1, shuffle=False, num_workers=0
    )
    for i, item in tqdm(enumerate(dataloader), total=len(dataloader)):
        qa_prompt, mcqa_prompt, cloze_prompt = item
        # print(f"QA Prompt: {qa_prompt}")
        # print(f"MCQA Prompt: {mcqa_prompt}")
        # print(f"Cloze Prompt: {cloze_prompt}")
        # print("\n")

        prompt = qa_prompt[0][0]
        label = qa_prompt[1][0]
        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        tokenized_prompt = {k: v.to(device) for k, v in tokenized_prompt.items()}

        tokenized_label = tokenizer(label, return_tensors="pt", padding=True)
        with torch.no_grad():
            predictions = model.generate(
                **tokenized_prompt,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=tokenized_label.input_ids.shape[-1],
            )
        generated_answer_text_qa = tokenizer.decode(
            predictions[0][len(tokenized_prompt["input_ids"][0]) :]
        )

        prompt = mcqa_prompt[0][0]
        label = mcqa_prompt[1][0]
        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        tokenized_prompt = {k: v.to(device) for k, v in tokenized_prompt.items()}
        with torch.no_grad():
            outputs = model(**tokenized_prompt)
            generated_logits = outputs.logits

        option_ids = [tokenizer.encode(f" {chr(65 + i)}")[-1] for i in range(4)]
        option_logits = generated_logits[:, -1, option_ids]
        generated_choice_mcqa = f"{chr(65 + torch.argmax(option_logits).item())}"

        prompt = cloze_prompt[0][0]
        label = cloze_prompt[1][0]
        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        tokenized_prompt = {k: v.to(device) for k, v in tokenized_prompt.items()}
        tokenized_label = tokenizer(label, return_tensors="pt", padding=True)
        with torch.no_grad():
            predictions = model.generate(
                **tokenized_prompt,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=tokenized_label.input_ids.shape[-1],
            )
        generated_answer_text_cloze = tokenizer.decode(
            predictions[0][len(tokenized_prompt["input_ids"][0]) :]
        )

        # print(f"QA Generated Answer: {generated_answer_text_qa}")
        # print(f"MCQA Generated Answer: {generated_choice_mcqa}")
        # print(f"Cloze Generated Answer: {generated_answer_text_cloze}")
        generation_tracker.update(
            qa_prompt=qa_prompt[0][0],
            qa_label=qa_prompt[1][0],
            mcqa_prompt=mcqa_prompt[0][0],
            mcqa_label=mcqa_prompt[1][0],
            cloze_prompt=cloze_prompt[0][0],
            cloze_label=cloze_prompt[1][0],
            qa_generated_text=generated_answer_text_qa,
            mcqa_generated_text=generated_choice_mcqa,
            cloze_generated_text=generated_answer_text_cloze,
        )

    generation_tracker.save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reformat_type",
        type=str,
        choices=[
            "mcqa",
            "cloze",
            "odd-one-out",
            "analogy-mcqa",
            "comprehension-qa",
            "comprehension-mcqa",
        ],
        default="mcqa",
    )
    parser.add_argument("--model_family", type=str, default="phi")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--num_options", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_on_all", action="store_true")
    parser.add_argument("--reinitialize_weights", action="store_true")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--qualitative_analysis", action="store_true")

    args = parser.parse_args()
    for key, value in vars(args).items():
        print(key, ":", value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # throw error if model_path is not provided
    if args.model_path == "" and args.use_pretrained == False:
        print("Please provide a model_path")
        sys.exit()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    model.to(device)
    if args.qualitative_analysis:
        qualitative_analysis(model, tokenizer, args, device)
    else:
        success_rate = evaluate_prompt(model, tokenizer, args, device)
        if True:
            print(f"Success Rate {args.dataset_path}:", success_rate)
            # log the model_path, dataset_path, success rate to a file, create the file with date and time if it doesn't exist
            with open(f"./logs/evaluation_results_{args.reformat_type}.csv", "a") as f:
                if args.reformat_type == "mcqa":
                    f.write(
                        f"{args.model_path},{args.dataset_path},{success_rate},{args.reformat_type},{args.num_options}\n"
                    )
                elif args.reformat_type == "cloze":
                    f.write(
                        f"{args.model_path},{args.dataset_path},{success_rate},{args.reformat_type}\n"
                    )
                elif args.reformat_type == "comprehension-mcqa":
                    f.write(
                        f"{args.model_path},{args.dataset_path},{success_rate},{args.reformat_type},{args.num_options}\n"
                    )
                elif args.reformat_type == "comprehension-qa":
                    f.write(
                        f"{args.model_path},{args.dataset_path},{success_rate},{args.reformat_type}\n"
                    )
                elif args.reformat_type == "analogy-mcqa":
                    f.write(
                        f"{args.model_path},{args.dataset_path},{success_rate},{args.reformat_type},{args.num_options}\n"
                    )
                elif args.reformat_type == "odd-one-out":
                    f.write(
                        f"{args.model_path},{args.dataset_path},{success_rate},{args.reformat_type},{args.num_options}\n"
                    )
