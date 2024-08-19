# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import json
import pandas as pd
from transformers import AutoTokenizer

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset


class GroundingDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):

        if partition == "train":
            file_path = dataset_config.train_data_path

        elif partition == "validation":
            file_path = dataset_config.test_data_path

        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            self.annotated_data = [json.loads(line) for line in f]
        # self.annotated_data = self.annotated_data[:100]

    def __len__(self):
        return len(self.annotated_data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        prompt = self.annotated_data[index]["prompt"]

        if not prompt.startswith("<|begin_of_text|><|start_header_id|>user<|end_header_id|>"):
            prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|> " + prompt
        
        if 
        
        # if not self.tokenizer.encode(prompt)[0] == self.tokenizer.bos_token_id:
        #     prompt = self.tokenizer.bos_token + prompt
        
        # encoded = prompt + self.annotated_data[index]['completion']
        # prompt = torch.tensor(
        #     self.tokenizer.encode(prompt), dtype=torch.int64
        # )
        # encoded = self.tokenizer.encode(encoded)
        if index == 0:  # perform a validation check on the first encoded
            assert "prompt" in self.annotated_data[index]
            assert "completion" in self.annotated_data[index]
            assert self.annotated_data[index]["prompt"].startswith("<|begin_of_text|><|start_header_id|>")
            assert self.annotated_data[index]["text"].endswith("<|eot_id|>")
            
        encoded = self.tokenizer.encode(self.annotated_data[index]["text"])
        
        assert encoded[0] == self.tokenizer.bos_token_id, "BOS token not correctly added"
        assert encoded[-1] == self.tokenizer.eos_token_id, "EOS token not correctly added"

        # if encoded[0] != self.tokenizer.bos_token_id:
        #     encoded = [self.tokenizer.bos_token_id] + encoded
        # encoded.append(self.tokenizer.eos_token_id)
        encoded = torch.tensor(
            encoded, dtype=torch.int64
        )
        labels = copy.deepcopy(encoded)
        labels[: len(prompt)] = -1
        encoded_mask = encoded.ge(0)
        label_mask = labels.ge(0)
        encoded[~encoded_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": encoded.tolist(),
            "labels": labels.tolist(),
            "attention_mask": encoded_mask.tolist(),
        }
