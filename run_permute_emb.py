# Script for permuting generated embeddings of mutational datasets

import os
import sys
import tables
import pandas as pd
import numpy as np
import json
import argparse

from tqdm import tqdm
from datetime import datetime

from src.encoding.encoding_classes import get_emb_info
from src.params.emb import (
    PROTXLSTM_INFO,
    CARP_INFO,
    TRANSFORMER_INFO,
    PROTMAMBA_INFO,
    POET_INFO,
)
from src.utils import checkNgen_folder
from src.params.sys import RAND_SEED

parser = argparse.ArgumentParser(description="Permute generated Embedding")


parser.add_argument(
    "--ref_split_path",
    type=str,
    metavar="P",
    help="full path to the dataset, in pkl or panda readable format, \
    ie: data/proeng/gb1/two_vs_rest.pkl or data/annotation/scl/balanced.csv",
)

parser.add_argument(
    "--goal_split_path",
    type=str,
    metavar="P",
    help="full path to the dataset, in pkl or panda readable format, \
    ie: data/proeng/gb1/sampled.pkl or data/annotation/scl/balanced.csv",
)

parser.add_argument(
    "--dataset_folder",
    type=str,
    metavar="O",
    help="the folder structure of dataset ie: progen/gb1/sampled or annotation/scl/balanced",
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
    "--flatten_emb",
    metavar="FE",
    default=False,
    help="if (False) and how ('mean', 'max']) to flatten the embedding (default: False)",
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
    "--sampling_seed",
    type=int,
    default=RAND_SEED,
    help="seed for sampling sequences within the MSA (default: 42)",
)


def main(args):

    folder = args.embed_folder
    if args.resample_param:
        folder = args.embed_folder + "-stat"
    elif args.reset_param:
        folder = args.embed_folder + "-rand"
        if isinstance(args.reset_param, str):
            folder += f"-{args.reset_param}"
    elif args.checkpoint != 1.0:
        folder = args.embed_folder + f"-{args.checkpoint}"

    curr_split, _ = os.path.splitext(os.path.basename(args.goal_split_path))

    _, _, total_emb_layer = get_emb_info(args.encoder_name)

    if args.encoder_name in PROTXLSTM_INFO.keys():
        info = PROTXLSTM_INFO
    if args.encoder_name in TRANSFORMER_INFO.keys():
        info = TRANSFORMER_INFO
    if args.encoder_name in CARP_INFO.keys():
        info = CARP_INFO
    if args.encoder_name in PROTMAMBA_INFO.keys():
        info = PROTMAMBA_INFO
    if args.encoder_name in POET_INFO.keys():
        info = POET_INFO
    embed_dim, *_ = info[args.encoder_name]

    # reference_split_path = 'data/proeng/gb1_trunc/sampled_trunc.csv'
    ref_file = pd.read_csv(args.ref_split_path)

    # curr_split_path = f'data/proeng/gb1_trunc/{curr_split}_trunc.csv'
    goal_file = pd.read_csv(args.goal_split_path)

    assert (
        ref_file.sequence.str.len().max() == goal_file.sequence.str.len().max()
    ), "split files contain different sequence lengths!"

    init_array_list = [None] * total_emb_layer

    embeddings = {}
    new_paths = {}

    for split in args.subset_list:

        use_sampling_seed = (
            "protxlstm" in args.encoder_name
            or "protmamba" in args.encoder_name
            or "poet" in args.encoder_name
        )

        earray_dim = (0, embed_dim)
        path = f'{folder}/seed-{args.embed_torch_seed}/{"sampling_seed-" + str(args.sampling_seed) + "/" if use_sampling_seed else ""}{args.dataset_folder}/{args.encoder_name}/{args.flatten_emb}/{split}/embedding.h5'
        if not os.path.exists(path):
            print(f"Does not exist: {path}")
            raise FileNotFoundError(path)

        print(f"Load all layers from {path}...")
        emb_table = tables.open_file(path)
        emb_table.flush()

        layers = {}
        for layer in range(total_emb_layer):
            layers[f"layer{layer}"] = getattr(emb_table.root, "layer" + str(layer))[
                :
            ]

        emb_table.close()

        new_paths[split] = os.path.join(
            *(path.split("/")[:-5] + [curr_split] + path.split("/")[-4:-1])
        )
        embeddings[split] = layers

    _ = [checkNgen_folder(p) for (k, p) in new_paths.items()]

    files = {}
    for split in args.subset_list:
        filepath = os.path.join(new_paths[split], "embedding.h5")
        if os.path.exists(filepath):
            print(f"Overwriting file: {filepath}")
            os.remove(filepath)

        files[split] = tables.open_file(
            os.path.join(new_paths[split], "embedding.h5"), mode="a"
        )

    # initialize tables
    for split, f_split in files.items():
        print(f"Create file: {f_split.filename}")
        for emb_layer in range(total_emb_layer):
            init_array_list[emb_layer] = f_split.create_earray(
                f_split.root,
                "layer" + str(emb_layer),
                tables.Float32Atom(),
                earray_dim,
            )

    ref_split_idxs = {k: 0 for k in args.subset_list}
    curr_split_idxs = {k: 0 for k in args.subset_list}

    for (i, ref), (j, curr) in tqdm(
        zip(ref_file.iterrows(), goal_file.iterrows()),
        total=len(ref_file),
        desc="Save permuted embedding",
    ):
        assert ref.sequence == curr.sequence, "sequences do not match!"
        from_set = ref.set if np.isnan(ref.validation) else "val"
        to_set = curr.set if np.isnan(curr.validation) else "val"
        for layer in range(total_emb_layer):
            getattr(files[to_set].root, "layer" + str(layer)).append(
                np.expand_dims(
                    embeddings[from_set][f"layer{layer}"][ref_split_idxs[from_set]],
                    axis=0,
                )
            )
        ref_split_idxs[from_set] += 1
        curr_split_idxs[to_set] += 1

    for _, f in files.items():
        f.close()

    print("Finished permuting embedding.")


if __name__ == "__main__":
    args = parser.parse_args()

    log_folder = checkNgen_folder("logs/run_permute_emb")

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

    log_dets = "{}2{}|{}|{}-{}|{}-{}".format(
        os.path.basename(os.path.splitext(args.ref_split_path)[0]),
        os.path.basename(os.path.splitext(args.goal_split_path)[0]),
        args.encoder_name,
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

    main(args)

    f.close()
