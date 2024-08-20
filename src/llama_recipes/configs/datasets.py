# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    trust_remote_code: bool = False


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""
    
@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class vitaminc_dataset:
    dataset: str = "vitaminc"
    train_split: str = "train"
    test_split: str = "validation"
    train_data_path: str = "/lustre/fsw/portfolios/llmservice/users/prasoonv/workspace/grounding-guardrails/training_data/vitc_train_10k.jsonl"
    test_data_path: str = "/lustre/fsw/portfolios/llmservice/users/prasoonv/workspace/grounding-guardrails/training_data/vitc_val_1k.jsonl"


@dataclass
class ragtruth_dataset:
    dataset: str = "ragtruth"
    train_split: str = "train"
    test_split: str = "validation"
    train_data_path: str = "/lustre/fsw/portfolios/llmservice/users/prasoonv/workspace/grounding-guardrails/training_data/ragtruth_train_10k.jsonl"
    test_data_path: str = "/lustre/fsw/portfolios/llmservice/users/prasoonv/workspace/grounding-guardrails/training_data/ragtruth_val_1k.jsonl"


@dataclass
class faithfulness_blended_dataset:
    dataset: str = "faithfulness_blended"
    train_split: str = "train"
    test_split: str = "validation"
    train_data_path: str = "/lustre/fsw/portfolios/llmservice/users/prasoonv/workspace/grounding-guardrails/training_data/blended_train_10k.jsonl"
    test_data_path: str = "/lustre/fsw/portfolios/llmservice/users/prasoonv/workspace/grounding-guardrails/training_data/blended_val_1k.jsonl"
