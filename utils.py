import json
import os
import torch


import yaml
import copy
import numpy as np
import torch.nn as nn

from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


def get_model_identifiers_from_yaml(model_family):
    # path is model_configs.yaml
    """
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    """
    model_configs = {}
    with open("./config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def load_model_and_tokenizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if args.model_name == "EleutherAI/gpt-neo-125M":
    #     model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    #     tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    # elif args.model_name == "EleutherAI/gpt-neo-1.3B":
    #     model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
    #     tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    # elif args.model_name == "EleutherAI/gpt-neo-2.7B":
    #     model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
    #     tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # elif args.model_name == "EleutherAI/gpt-j-6B":
    #     # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
    #     model = GPTJForCausalLM.from_pretrained(
    #         "EleutherAI/gpt-j-6B",
    #     ).to(device)
    #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # elif args.model_name == "NousResearch/Llama-2-7b-chat-hf":
    #     PATH_TO_CONVERTED_WEIGHTS = "NousResearch/Llama-2-7b-chat-hf"
    #     model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(device)
    #     # model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(device)
    #     tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    # elif args.model_name == "mistralai/Mistral-7B-v0.1":
    #     PATH_TO_CONVERTED_WEIGHTS = "mistralai/Mistral-7B-v0.1"
    #     # # model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(device)
    #     model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(
    #         device
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    model_cfg = get_model_identifiers_from_yaml(args.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token

    model = None
    config = AutoConfig.from_pretrained(model_id)
    for attempt in range(3):
        try:
            # do thing
            if args.use_pretrained:
                args.method_name = "pretrained"
                args.checkpoint_idx = 0
                print(f"Loading pretrained from {model_id}")
                # model = AutoModelForCausalLM.from_pretrained(
                #     model_id,
                #     config=config,
                #     use_flash_attention_2=model_cfg["flash_attention2"] == "true",
                #     torch_dtype=torch.bfloat16,
                #     trust_remote_code=True,
                # )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    use_flash_attention_2=model_cfg["flash_attention2"] == "true",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )

            else:
                args.method_name = "fine-tuned-retain95"
                args.checkpoint_idx = 0
                print(f"Loading checkpoint from {args.model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    config=config,
                    use_flash_attention_2=model_cfg["flash_attention2"] == "true",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
        except Exception as e:
            print(e)
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")

    model = model.eval()

    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    if args.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)

    return model, tokenizer
