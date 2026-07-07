# Copyright (c) 2025 Antonia Ebner
# Description:
#     Short runfile script to preprocess downloaded data for the project.


import os
import argparse
import shutil

import pandas as pd


def main(args):

    print(f"Preprocessing data found in: {args.data_folder}")

    # Rename wildtype fasta files to "wildtype.fasta" and rename label of wildtype fasta files to >seq0
    for orig_path in [
        os.path.join(args.data_folder, "proeng", "aav", "P03135.fasta"),
        os.path.join(args.data_folder, "proeng", "gb1", "5LDE_1.fasta"),
    ]:
        file_path = os.path.join(os.path.dirname(orig_path), "wildtype.fasta")
        print(f"Create wildtype fasta file: {file_path}")
        with open(orig_path, "r") as f_ref, open(file_path, "a") as f_wildtype:
            for i, line in enumerate(f_ref.readlines()):
                if i == 0:
                    line = ">seq0\n"
                elif "gb1" in orig_path:
                    line = line[:56] + "\n"

                f_wildtype.write(line)

    # Truncate gb1 splits to 56
    os.makedirs(os.path.join(args.data_folder, "proeng", "gb1_trunc"))
    for split in ["sampled", "low_vs_high", "two_vs_rest"]:
        path = os.path.join(args.data_folder, "proeng", "gb1", split + ".csv")
        print(f"Truncate sequences in: {path}")
        df = pd.read_csv(path)
        df.sequence = df.sequence.str.slice(0, 56)
        new_path = os.path.join(args.data_folder, "proeng", "gb1_trunc", split)
        shutil.copyfile(path, new_path + ".csv")
        df.to_csv(new_path + "_trunc.csv", index=False)

    # only keep AAV sequences used in the splits
    for split in ["one_vs_many", "two_vs_many"]:
        path = os.path.join(args.data_folder, "proeng", "aav", split + ".csv")
        print(f"Truncate file: {path}")
        df = pd.read_csv(path)
        df = df.dropna(subset=["set"])
        df.to_csv(path, index=False)

    # create fasta files for non-mutational datasets
    task2split = {
        "structure": ["secondary_structure/tape_ss3_processed.csv"],
        "proeng": ["thermo/mixed_split.csv"],
        "annotation": ["scl/balanced.csv"],
    }

    for task in ["structure", "proeng", "annotation"]:
        for split in task2split[task]:
            path = os.path.join(args.data_folder, task, split)
            fasta_path = os.path.splitext(path)[0] + ".fasta"
            df = pd.read_csv(path)
            print(f"Create fasta file: {fasta_path}")
            with open(fasta_path, "a") as f:
                for i, seq in enumerate(df.sequence):
                    f.write(f">seq{str(i)}\n{seq}\n")

    task2split = {
        "structure": ["secondary_structure/tape_ss3_processed.csv"],
        "proeng": [
            "gb1_trunc/sampled.csv",
            "aav/two_vs_many.csv",
            "thermo/mixed_split.csv",
        ],
        "annotation": ["scl/balanced.csv"],
    }

    # create split files
    for task in ["structure", "proeng", "annotation"]:
        for split in task2split[task]:
            path = os.path.join(args.data_folder, task, split)
            df = pd.read_csv(path)
            df["ID"] = [f"seq{i}" for i in range(len(df))]

            masks = {
                "train": (df.set == "train") & (df.validation != True),
                "val": (df.set == "train") & (df.validation == True),
            }

            if task == "structure":
                testing_masks = {
                    "casp12": (df.set == "casp12"),
                    "ts115": (df.set == "ts115"),
                    "cb513": (df.set == "cb513"),
                }
            else:
                testing_masks = {"test": (df.set == "test")}

            masks.update(testing_masks)

            for subset, mask in masks.items():
                ids = df.ID.where(mask).dropna().reset_index(drop=True)
                filename = os.path.splitext(path)[0] + f"_{subset}_ids.csv"
                ids.to_csv(filename, header=False)
                print(f"Create subset_id file: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing")

    parser.add_argument(
        "--data_folder",
        type=str,
        default="./data",
        help="full path to the dataset folder",
    )

    args = parser.parse_args()
    main(args)

    print("Preprocessing finished.")
