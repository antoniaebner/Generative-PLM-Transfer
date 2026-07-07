import argparse

import os

import pandas as pd

from src.preprocess.data import process_data

GB1_SPLITS = ["sampled", "low_vs_high", "two_vs_rest"]
AAV_SPLITS = ["two_vs_many", "one_vs_many"]
SS3_SPLITS = ["tape_ss3_processed"]  #'casp12', 'cb513', 'ts115',
THERMO_SPLITS = ["mixed_split"]
ANNOTATION_SPLITS = ["balanced"]
RH_SPLITS = ["tape_rh_processed"]

DSET_TO_SPLITS = {
    "gb1": GB1_SPLITS,
    "gb1_trunc": GB1_SPLITS,
    "aav": AAV_SPLITS,
    "tape_ss3_processed": SS3_SPLITS,
    "thermo": THERMO_SPLITS,
    "scl": ANNOTATION_SPLITS,
    "remote_homology": RH_SPLITS,
}

parser = argparse.ArgumentParser(description="Protein transfer with pytorch models")
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
    "--a3m_path",
    type=str,
    default="a3m/uniclust30.a3m",
    help="path leading to the needed a3m files, ie: a3m/uniclust30.a3m",
)

parser.add_argument(
    "--filename",
    type=str,
    default="open_protein_set_memmap",
    help="desired filename to save the processed msas as, ie: open_protein_set_memmap",
)

parser.add_argument(
    "--is_mutation",
    type=int,
    default=0,
    help="true if dataset is a mutation dataset and individual sequences are highly similar, ie: 1",
)

parser.add_argument(
    "--calc_similarity",
    type=int,
    default=1,
    help="true if MSA similarity and weights should be calculated, ie: 1",
)

parser.add_argument(
    "--skip_processing",
    type=int,
    default=0,
    help="true if MSA similarity and weights should be calculated, ie: 1",
)


def main(args):
    if not args.skip_processing:
        process_data(args)
        print("finished processing data")

    # create split files
    root = os.path.join(*(["data"] + args.output_dir.strip("/").split("/")[2:]))
    for dset in DSET_TO_SPLITS[os.path.basename(root)]:
        if os.path.basename(root) == "tape_ss3_processed":
            path = root + ".csv"
        else:
            path = os.path.join(root, dset + ".csv")
        df = pd.read_csv(path)

        df["ID"] = [f"seq{i}" for i in range(len(df))]

        masks = {
            "train": (df.set == "train") & (df.validation != True),
            "val": (df.set == "train") & (df.validation == True),
        }

        if os.path.basename(root) == "tape_ss3_processed":
            testing_masks = {
                "casp12": (df.set == "casp12"),
                "ts115": (df.set == "ts115"),
                "cb513": (df.set == "cb513"),
            }
        elif os.path.basename(root) == "remote_homology":
            testing_masks = {
                "test_family_holdout": (df.set == "test_family_holdout"),
                "test_superfamily_holdout": (df.set == "test_superfamily_holdout"),
                "test_fold_holdout": (df.set == "test_fold_holdout"),
            }
        else:
            testing_masks = {"test": (df.set == "test")}

        masks.update(testing_masks)

        for split, mask in masks.items():
            ids = df.ID.where(mask).dropna().reset_index(drop=True)
            ids.to_csv(os.path.splitext(path)[0] + f"_{split}_ids.csv", header=False)

    # # truncate gb1 files
    # if os.path.basename(root) == 'gb1_trunc':
    #     for dset in DSET_TO_SPLITS('gb1_trunc'):
    #         path = os.path.join(root, dset + '.csv')
    #         df = pd.read_csv(path)
    #         df.sequence = df.sequence.str.slice(0, 56)
    #         path = os.path.splitext(path)[0] + '_trunc.csv'
    #         df.to_csv(path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
