# MIT License

# Copyright (c) Microsoft Corporation.

# Modified by Antonia Ebner, 2026:
#   - add Prot-xLSTM, ProtMamba, and PoET
#   - added additional arguments

"""Script for pre generating all embeddings"""


from __future__ import annotations

import os
import sys
import json
import argparse

from datetime import datetime

from src.encoding.gen_encoding import GenerateEmbeddings
from src.encoding.encoding_classes import seed_all
from src.params.emb import (
    PROTXLSTM_INFO,
    PROTMAMBA_INFO,
    POET_INFO,
)
from src.params.sys import RAND_SEED
from src.utils import checkNgen_folder, get_filename


parser = argparse.ArgumentParser(description="Embedding Generation")


parser.add_argument(
    "--dataset_path",
    type=str,
    metavar="P",
    help="full path to the dataset, in pkl or panda readable format, \
    ie: data/proeng/gb1/two_vs_rest.pkl or data/annotation/scl/balanced.csv",
)

parser.add_argument(
    "--encoder_name",
    type=str,
    metavar="EN",
    help="the name of the encoder, ie: esm1b_t33_650M_UR50S",
)

parser.add_argument(
    "--checkpoint",
    type=float,
    metavar="CP",
    default=1,
    help="the fraction of the pretrain model, ie: 0.5",
)

parser.add_argument(
    "--checkpoint_folder",
    type=str,
    metavar="CPF",
    default="pretrain_checkpoints/carp",
    help="the folder for the pretrain model, ie: pretrain_checkpoints/carp",
)

parser.add_argument(
    "--reset_param",
    type=str,
    metavar="RIP",
    default="",
    help="if update the full model to xavier_uniform_ (default: False)",
)

parser.add_argument(
    "--resample_param",
    type=bool,
    metavar="STP",
    default=False,
    help="if update the full model to xavier_normal_ (default: False)",
)

parser.add_argument(
    "--embed_torch_seed",
    type=int,
    metavar="ETS",
    default=RAND_SEED,
    help="the torch seed for random init and stat transfer (default: 42)",
)

parser.add_argument(
    "--embed_batch_size",
    type=int,
    metavar="EBS",
    default=1,
    help="the embedding batch size, set to 0 to encode all in a single batch (default: 1)",
)

parser.add_argument(
    "--flatten_emb",
    metavar="FE",
    default=False,
    help="if (False) and how ('mean', 'max']) to flatten the embedding (default: False)",
)

parser.add_argument(
    "--seq_start_idx",
    metavar="SSI",
    default=False,
    help="the index for the start of the sequence (default: False)",
)

parser.add_argument(
    "--seq_end_idx",
    metavar="SEI",
    default=False,
    help="the index for the end of the sequence (default: False)",
)

parser.add_argument(
    "--sort_context",
    type=bool,
    metavar="STP",
    default=False,
    help="if use a sorted context (default: False)",
)

parser.add_argument(
    "--subset_list",
    metavar="SL",
    type=json.loads,
    default=["train", "val", "test"],
    help="the index for the end of the sequence (default: False)",
)

parser.add_argument(
    "--embed_folder",
    type=str,
    default="embeddings",
    metavar="O",
    help="the parent folder for embeddings (default: 'embeddings')",
)

parser.add_argument(
    "--yield_no_context",
    type=bool,
    default=False,
    help="whether the dataloader should yield sequences with their context appended",
)

parser.add_argument(
    "--is_mutation",
    type=bool,
    default=False,
    help="whether the dataset is a mutation dataset",
)

parser.add_argument(
    "--max_context_length",
    type=int,
    default=-1,
    help="maximum context length allowed (default: -1)",
)

parser.add_argument(
    "--max_context_sequences",
    type=int,
    default=200,
    help="max number of context sequences to allow in the context length limit (default: 200)",
)

parser.add_argument(
    "--max_similarity",
    type=float,
    default=0.98,
    help="max sequence identity allowed in the context of an MSA (default: 0.98)",
)

parser.add_argument(
    "--max_dissimilarity",
    type=float,
    default=0.7,
    help="max sequence dissimilarity allowed in the context of an MSA (default: 0.7)",
)

parser.add_argument(
    "--mlstm_chunksize",
    type=int,
    default=1024,
)

parser.add_argument(
    "--chunk_chunk_size",
    type=int,
    default=2**13,
    help="max sequence dissimilarity allowed in the context of an MSA (default: 0.7)",
)

parser.add_argument(
    "--sampling_seed",
    type=int,
    default=RAND_SEED,
    help="seed for sampling sequences within the MSA (default: 42)",
)

parser.add_argument(
    "--msa_path",
    type=str,
    default=None,
    help="path to the MSA memmap file (default: None)",
)

parser.add_argument(
    "--manual_layer_min",
    metavar="LMIN",
    default=False,
    help="the number of layer for manual start range (default: False)",
)

parser.add_argument(
    "--manual_layer_max",
    metavar="LMAX",
    default=False,
    help="the number of layer for manual end range (default: False)",
)

parser.add_argument(
    "--max_seq_len",
    type=int,
    default=2048,
    help="max number of amino acids allowed per sequence (default: 1022)",
)

parser.add_argument(
    "--overwrite_file",
    type=int,
    default=1,
    help="if overwrite existing embedding file (default: False)",
)

# TODO add encoder_params

args = parser.parse_args()

log_folder = checkNgen_folder("logs/run_pregen_emb")

args.reset_param = (
    bool(args.reset_param) if args.reset_param in ["True", ""] else args.reset_param
)

if args.reset_param:
    randorinit = f"rand-{args.reset_param}"
elif args.resample_param:
    randorinit = "stat"
else:
    randorinit = "none"

if (
    args.encoder_name in list(PROTXLSTM_INFO.keys())
    or args.encoder_name in list(PROTMAMBA_INFO.keys())
    or args.encoder_name in list(POET_INFO.keys())
):
    use_msa = "msa"
else:
    use_msa = "none"

log_dets = "{}-{}|{}|{}|{}-{}|{}-{}".format(
    get_filename(os.path.dirname(args.dataset_path)),
    get_filename(args.dataset_path),
    args.encoder_name,
    args.flatten_emb,
    randorinit,
    args.embed_torch_seed,
    use_msa,
    args.sampling_seed,
)

# log outputs
f = open(
    os.path.join(
        log_folder,
        "{}||{}.out".format(log_dets, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ),
    "w",
)
sys.stdout = f

seed_all(args.embed_torch_seed)

print(f"Arguments: {args}")

GenerateEmbeddings(
    dataset_path=args.dataset_path,
    encoder_name=args.encoder_name,
    checkpoint=args.checkpoint,
    checkpoint_folder=args.checkpoint_folder,
    reset_param=args.reset_param,
    resample_param=args.resample_param,
    embed_torch_seed=args.embed_torch_seed,
    embed_batch_size=args.embed_batch_size,
    flatten_emb=args.flatten_emb,
    seq_start_idx=args.seq_start_idx,
    seq_end_idx=args.seq_end_idx,
    subset_list=args.subset_list,
    embed_folder=checkNgen_folder(args.embed_folder),
    max_context_length=args.max_context_length,
    max_context_sequences=args.max_context_sequences,
    max_similarity=args.max_similarity,
    max_dissimilarity=args.max_dissimilarity,
    yield_with_context=not bool(args.yield_no_context),
    is_mutation=bool(args.is_mutation),
    chunk_chunk_size=args.chunk_chunk_size,
    mlstm_chunksize=args.mlstm_chunksize,
    sampling_seed=args.sampling_seed,
    msa_path=args.msa_path,
    manual_layer_min=args.manual_layer_min,
    manual_layer_max=args.manual_layer_max,
    max_seq_len=args.max_seq_len,
    overwrite_file=bool(args.overwrite_file),
    sort_context=args.sort_context,
)

f.write("Embedding generation finished successfully.")

f.close()
