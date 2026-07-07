# Original code from Prot-xLSTM under Apache License 2.0.
#
# Modifications made by Antonia Ebner
#   - Add argument parsing
#   - Using directory of a3m files instead of 1 big a3m file
#   - Include pre-calculation of MSA weights and similarities

import sys
import os
import csv
import argparse

import numpy as np
from tqdm import tqdm

from src.preprocess.data import process_msa


def main(args):
    msa_paths_from_file = {
        os.path.splitext(k)[0]: os.path.join(args.data_dir, k)
        for k in os.listdir(args.data_dir)
        if os.path.isfile(os.path.join(args.data_dir, k)) and "seq" in k
    }
    msa_items = [
        (f"seq{str(i)}", msa_paths_from_file[f"seq{str(i)}"])
        for i in range(len(msa_paths_from_file))
    ]

    token_dictionary = {}
    total_length_tokens = 0
    meta_dictionary = {}
    total_length_meta = 0

    # First pass: calculate total length of all concatenated arrays
    for item in tqdm(msa_items):
        k, t, s, w = process_msa(
            item, args.use_wildtype_context, calc_similarity=args.calc_similarity
        )
        token_dictionary[k] = t
        total_length_tokens += len(t)
        meta_dictionary[k] = (s, w)
        total_length_meta += len(s)

    # Initialize the memmap array with the calculated total length
    memmap_path = os.path.join(args.output_dir, f"{args.filename}.dat")
    concatenated_array = np.memmap(
        memmap_path, dtype="int8", mode="w+", shape=(total_length_tokens,)
    )

    with open(
        f"{args.output_dir}/{args.filename}_indices.csv", "w", newline=""
    ) as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(["msa_id", "Start", "End"])

        start_index = 0
        for key, array in token_dictionary.items():
            end_index = start_index + len(array) - 1
            concatenated_array[start_index : end_index + 1] = array  # Write to memmap
            csvwriter.writerow([key, start_index, end_index])
            start_index = end_index + 1

    # Ensure the data is written to disk
    concatenated_array.flush()

    # Repeat above for properties as well
    memmap_path = os.path.join(args.output_dir, f"{args.filename}_similarity.dat")
    similarity_array = np.memmap(
        memmap_path, dtype="float32", mode="w+", shape=(total_length_meta,)
    )

    memmap_path = os.path.join(args.output_dir, f"{args.filename}_weights.dat")
    weights_array = np.memmap(
        memmap_path, dtype="float32", mode="w+", shape=(total_length_meta,)
    )

    with open(
        f"{args.output_dir}/{args.filename}_meta_indices.csv", "w", newline=""
    ) as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(["msa_id", "Start", "End"])

        start_index = 0
        for key, (sim, weights) in meta_dictionary.items():
            end_index = start_index + len(sim) - 1
            similarity_array[start_index : end_index + 1] = sim  # Write to memmap
            weights_array[start_index : end_index + 1] = weights  # Write to memmap
            csvwriter.writerow([key, start_index, end_index])
            start_index = end_index + 1

    # Ensure the data is written to disk
    similarity_array.flush()
    weights_array.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a3m files to use as tokenized memmap arrays for Prot-xLSTM."
    )

    parser.add_argument(
        "data_dir",
        type=str,
        help="path to the dataset directory containing MSAs, \
        ie: msas/individual_msas/proeng/gb1/a3m_files or msas/individual_msas/annotation/scl/a3m_files",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="path to the dataset directory, ie: msas/individual_msas/proeng/gb1/",
    )

    parser.add_argument(
        "--filename",
        type=str,
        help="filename without file extension, e.g. gb1_memmap",
    )

    parser.add_argument(
        "--use_wildtype_context",
        type=bool,
        default=False,
        help="true if dataset is a mutation dataset and individual sequences are highly similar, \
        so the context for the wildtype should be used, ie: 1",
    )

    parser.add_argument(
        "--calc_similarity",
        type=int,
        default=1,
        help="true if MSA similarity and weights should be calculated, ie: 1",
    )

    args = parser.parse_args()
    main(args)

    print("Finished processing data.")
