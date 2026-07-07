# MIT License

# Copyright (c) Microsoft Corporation.

# Modified by Antonia Ebner, 2026:
#   - add Prot-xLSTM, ProtMamba, and PoET
#   - added additional arguments
# 
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime

from src.params.sys import DEVICE, RAND_SEED
from src.probing.run_pytorch import Run_Pytorch
from src.utils import checkNgen_folder, get_filename
from src.params.emb import PROTXLSTM_INFO

parser = argparse.ArgumentParser(description="Protein transfer with pytorch models")

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
    default=128,
    help="the embedding batch size, set to 0 to encode all in a single batch (default: 128)",
)

parser.add_argument(
    "--flatten_emb",
    metavar="FE",
    default=False,
    help="if (False) and how ('mean', 'max']) to flatten the embedding (default: 'mean')",
)

parser.add_argument(
    "--embed_folder",
    metavar="EP",
    default=None,
    help="path to presaved embedding (default: None)",
)

parser.add_argument(
    "--test_embed_folder",
    default=None,
    help="path to presaved embedding used for test set. If None, will use path passed in ```embed_folder``` (default: None)",
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
    "--loader_batch_size",
    type=int,
    metavar="LBS",
    default=64,
    help="the batch size for train, val, and test dataloader (default: False)",
)

parser.add_argument(
    "--worker_seed",
    type=int,
    metavar="WS",
    default=RAND_SEED,
    help="the seed for dataloader (default: RAND_SEED)",
)

parser.add_argument(
    "--if_encode_all",
    type=bool,
    metavar="EA",
    default=False,
    help="if encode full dataset all layers on the fly (default: False)",
)

parser.add_argument(
    "--if_rerun_layer",
    type=bool,
    metavar="irl",
    default=False,
    help="if re run layers if already exist (default: False)",
)

parser.add_argument(
    "--if_multiprocess",
    type=bool,
    metavar="MP",
    default=False,
    help="if running all layers in parallel (default: False)",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    metavar="LR",
    default=1e-4,
    help="learning rate (default: 1e-4)",
)

parser.add_argument(
    "--lr_decay",
    type=float,
    metavar="LRD",
    default=0.1,
    help="factor by which to decay learning rate on plateau (default: 0.1)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    metavar="N",
    help="number of epochs to train (default: 20)",
)

parser.add_argument(
    "--early_stop",
    type=bool,
    default=True,
    metavar="ES",
    help="if initate early stopping (default: True)",
)

parser.add_argument(
    "--tolerance",
    type=int,
    default=10,
    metavar="T",
    help="tolerance for early stopping (default: 10)",
)

parser.add_argument(
    "--min_epoch",
    type=int,
    default=5,
    metavar="ME",
    help="minimal number of epochs for early stopping (default: 5)",
)

parser.add_argument(
    "--device",
    default=DEVICE,
    metavar="D",
    help="torch device (default: DEVICE)",
)

parser.add_argument(
    "--all_plot_folder",
    type=str,
    default="results/learning_curves",
    metavar="LC",
    help="the parent folder for all learning curves (default: 'results/learning_curves')",
)

parser.add_argument(
    "--all_result_folder",
    type=str,
    default="results/pytorch",
    metavar="O",
    help="the parent folder for all results (default: 'results/pytorch')",
)

parser.add_argument(
    "--sampling_seed",
    type=int,
    default=RAND_SEED,
    help="seed for sampling sequences within the MSA (default: 42)",
)

parser.add_argument(
    "--yield_with_context",
    default=True,
    help="whether the dataloader should yield sequences with their context appended",
)

parser.add_argument(
    "--is_mutation",
    default=False,
    help="whether the dataset is a mutation dataset",
)

parser.add_argument(
    "--max_context_length",
    type=int,
    default=200_000,
    help="maximum context length allowed (default: 200_000)",
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
    default=2**14,
    help="max sequence dissimilarity allowed in the context of an MSA (default: 0.7)",
)

parser.add_argument(
    "--msa_path",
    type=str,
    default=None,
    help="path to the MSA memmap file (default: None)",
)

parser.add_argument(
    "--acc_steps",
    type=int,
    default=1,
    help="number of gradient accumulation steps used during training (default: 1)",
)

# TODO add encoder_params

args = parser.parse_args()


log_folder = checkNgen_folder("logs/run_protran_pytorch")

args.reset_param = (
    bool(args.reset_param) if args.reset_param in ["True", ""] else args.reset_param
)

if args.reset_param:
    randorinit = f"rand-{args.reset_param}"
elif args.resample_param:
    randorinit = "stat"
else:
    randorinit = "none"

if args.encoder_name in list(PROTXLSTM_INFO.keys()):
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

print(f"Arguments: {args}")

Run_Pytorch(
    dataset_path=args.dataset_path,
    encoder_name=args.encoder_name,
    checkpoint=args.checkpoint,
    checkpoint_folder=args.checkpoint_folder,
    reset_param=args.reset_param,
    resample_param=args.resample_param,
    embed_torch_seed=args.embed_torch_seed,
    embed_batch_size=args.embed_batch_size,
    flatten_emb=args.flatten_emb,
    embed_folder=args.embed_folder,
    test_embed_folder=(
        args.test_embed_folder
        if args.test_embed_folder is not None
        else args.embed_folder
    ),
    seq_start_idx=args.seq_start_idx,
    seq_end_idx=args.seq_end_idx,
    manual_layer_min=args.manual_layer_min,
    manual_layer_max=args.manual_layer_max,
    loader_batch_size=args.loader_batch_size,
    worker_seed=args.worker_seed,
    if_rerun_layer=args.if_rerun_layer,
    if_encode_all=args.if_encode_all,
    if_multiprocess=args.if_multiprocess,
    learning_rate=args.learning_rate,
    lr_decay=args.lr_decay,
    epochs=args.epochs,
    early_stop=args.early_stop,
    tolerance=args.tolerance,
    min_epoch=args.min_epoch,
    device=args.device,
    all_plot_folder=args.all_plot_folder,
    all_result_folder=args.all_result_folder,
    sampling_seed=args.sampling_seed,
    msa_path=args.msa_path,
    is_mutation=args.is_mutation,
    yield_with_context=args.yield_with_context,
    max_context_length=args.max_context_length,
    max_context_sequences=args.max_context_sequences,
    max_similarity=args.max_similarity,
    max_dissimilarity=args.max_dissimilarity,
    chunk_chunk_size=args.chunk_chunk_size,
    mlstm_chunksize=args.mlstm_chunksize,
    acc_steps=args.acc_steps,
    # **encoder_params,
)

f.write("Experiments finished successfully.")

f.close()
