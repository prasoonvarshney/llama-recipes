# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import json
from transformers import AutoTokenizer

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import torch
from torch.utils.data import Dataset


class GenericSFTDataset(Dataset):
    """
    Class to load any generic LLM SFT dataset into a PyTorch Dataset.
    Note:
        - This class expects a tokenizer of AutoTokenizer type, with eos and bos tokens set.
        - This class expects the dataset to be in json format with at least two columns 'prompt' and 'completion'.
        - When generating the encoded text, the prompt and the completion are concatenated together.
        - When generating the labels, the prompt portion is masked out so the model doesn't learn the prompt structure / 
        doesn't backpropagate errors for those tokens.
    """
    def __init__(self, dataset_config, tokenizer: AutoTokenizer, partition="train"):

        if partition == "train":
            file_path = dataset_config.train_data_path

        elif partition == "validation" or partition == "test":
            file_path = dataset_config.test_data_path

        self.tokenizer = tokenizer
        with open(file_path, 'r') as f:
            if file_path.endswith('.jsonl'):
                self.annotated_data = [json.loads(line) for line in f]
            elif file_path.endswith('.json'):
                self.annotated_data = json.load(f)
            else:
                raise NotImplementedError(f"Dataloader not implemented for {file_path.split('.')[-1]} file format")
        # self.annotated_data = self.annotated_data[:100]

    def __len__(self):
        return len(self.annotated_data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        prompt = self.annotated_data[index]['prompt']
        completion = self.annotated_data[index]['completion']

        prompt = self.tokenizer.apply_chat_template([
            {
                "role": "user",
                "content": prompt.replace("[INST] ", "")  # TODO: Temporary, remove the replace (shouldn't exist in the dataset itself)
            }
        ], tokenize=False, add_generation_prompt=True)
        completion = completion + " " + self.tokenizer.eos_token
        prompt_encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_encoded = self.tokenizer.encode(completion, add_special_tokens=False)
        encoded = torch.cat([
            torch.tensor(prompt_encoded, dtype=torch.int64),
            torch.tensor(completion_encoded, dtype=torch.int64)
        ])
        if index == 0:  # perform a validation check on the first encoded
            # Llama 3.1 specific assertions
            assert prompt.startswith('<|begin_of_text|><|start_header_id|>')
            assert prompt.endswith("<|end_header_id|>\n\n")
            assert completion.endswith("<|eot_id|>")
            # General assertions
            assert encoded[0] == self.tokenizer.bos_token_id, "BOS token not correctly added"
            assert encoded[-1] == self.tokenizer.eos_token_id, "EOS token not correctly added"

        labels = copy.deepcopy(encoded)
        labels[: len(prompt_encoded)] = -1
        encoded_mask = encoded.ge(0)
        label_mask = labels.ge(0)
        encoded[~encoded_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        assert len(encoded) == len(labels), "Encoded and labels should have the same length"
        assert encoded[-1] == self.tokenizer.eos_token_id == labels[-1], "EOS token not correctly added"
        assert encoded[-3] == labels[-3], "Completion token should be the same in both encoded and labels"
        return {
            "input_ids": encoded.tolist(),
            "labels": labels.tolist(),
            "attention_mask": encoded_mask.tolist(),
        }
