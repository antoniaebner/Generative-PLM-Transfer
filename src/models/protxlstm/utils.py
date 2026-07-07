# Some of the objects in this file come from ProtMamba and mamba both under Apache License 2.0.

import json
import os

import numpy as np
import rich
import torch
from Bio import SeqIO
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
import wandb
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file

__all__ = ['AA_TO_ID', 'MASK_TO_ID', 'ID_TO_AA', 'load_model', 'encode_sequence', 'decode_sequence', 'clean_sequence', 'tokenizer',
           'load_sequences_from_msa_file', 'prepare_tokens', 'prepare_target', 'print_number_of_parameters',
           'print_config', 'print_zero_rank', 'is_zero_rank']

# Constants
AA_TO_ID = {'<cls>': 0,
            '<pad>': 1,
            '<eos>': 2,
            '<unk>': 3,
            'L': 4,
            'A': 5,
            'G': 6,
            'V': 7,
            'S': 8,
            'E': 9,
            'R': 10,
            'T': 11,
            'I': 12,
            'D': 13,
            'P': 14,
            'K': 15,
            'Q': 16,
            'N': 17,
            'F': 18,
            'Y': 19,
            'M': 20,
            'H': 21,
            'W': 22,
            'C': 23,
            'X': 24,
            'B': 25,
            'U': 26,
            'Z': 27,
            'O': 28,
            '.': 29,
            '-': 30,
            '<null_1>': 31,
            '<mask>': 32}

MASK_TO_ID = {"<mask-1>": 33,
              "<mask-2>": 34,
              "<mask-3>": 35,
              "<mask-4>": 36,
              "<mask-5>": 37,}

AA_TO_ID.update(MASK_TO_ID)

ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}

def is_zero_rank():
    return int(os.getenv('LOCAL_RANK', '0')) == 0

def print_zero_rank(var):
    if is_zero_rank():
        print(var)

def print_number_of_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_num_params = f"{num_params:_}"
    print("Number of trainable parameters: ", formatted_num_params)

# Sequence tools
def encode_sequence(sequence):
    """Tokenize a sequence of amino acids and add a cls token at the beginning."""
    tokenized_sequence = [AA_TO_ID[aa] if aa in AA_TO_ID else AA_TO_ID['<unk>'] for aa in sequence]
    return [AA_TO_ID['<cls>']] + tokenized_sequence

def decode_sequence(sequence):
    """Decode a sequence of tokens."""
    return "".join([ID_TO_AA[token] if token in ID_TO_AA else "<unk>" for token in sequence])

def clean_sequence(sequence):
    """Remove gaps and convert all residues to upper case."""
    return sequence.replace("-", "").upper()

def tokenizer(sequence_list, concatenate=True):
    """Tokenize a collection of sequences. If the sequences are aligned, the gaps will be removed
    and the insertions (lower case) will be promoted to upper case."""
    # clean and encode all sequences
    sequence_list = [encode_sequence(clean_sequence(sequence)) for sequence in sequence_list]
    if concatenate:
        # concatenate all sequences
        sequences = np.concatenate(sequence_list)
        # convert to tensor and add batch dimension
        return torch.asarray(sequences, dtype=torch.int8)[None,:]
    else:
        return [torch.asarray(sequence, dtype=torch.int8) for sequence in sequence_list]

def load_sequences_from_msa_file(file_path):
    """Load a collection of sequences from an a3m file."""
    with open(file_path, "r") as f:
        sequences = [str(record.seq) for record in SeqIO.parse(f, "fasta")]
    return sequences

# Others
def load_model(
    model_path,
    device,
    model_class,
    dtype=torch.bfloat16,
    **kwargs
):
    model = model_class.from_pretrained(
        model_path, device=device, dtype=dtype, **kwargs
    )
    return model

# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/hf.py
def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))

# https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/hf.py
def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict