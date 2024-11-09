from pathlib import Path
from datasets import load_dataset
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--config_name", type=str, required=True, help="Name of the config")
    args = parser.parse_args()
    
    dataset = load_dataset(args.hf_dataset, args.config_name)


    data_reformat_path = Path("data_reformat")
    config_path = data_reformat_path / args.config_name
    config_path.mkdir(parents=True, exist_ok=True)

    for key in dataset.keys():
        df = dataset[key].to_pandas()
        df.to_json(f"{config_path / key}.json", orient="records")

