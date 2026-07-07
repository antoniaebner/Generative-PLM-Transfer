# Original code from Protein-Transfer under MIT License.

# Copyright (c) Microsoft Corporation.

# Modified by Antonia Ebner, 2026:
#   - adapted to allow for multiple sampling seeds
# 
"""A script reorg results"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

from collections import defaultdict
from src.utils import checkNgen_folder
from src.analysis.utils import (
    metric_simplifier,
    STRUCT_TESTS,
    RH_TESTS,
    DEFAULT_AB_LIST,
    PRETRAIN_ARCH_LIST,
    DS_MODEL_LIST,
)
from src.analysis.perlayer import ResultParser
from src.params.emb import (
    CARP_MODEL_INFO,
    ESM_MODEL_INFO,
    PROTXLSTM_MODEL_INFO,
    PROTMAMBA_MODEL_INFO,
    POET_MODEL_INFO,
)


class ResultReorg:
    """A class for reorg layer wise results"""

    def __init__(
        self,
        layer_folder: str = "results",
        summary_folder: str = "results/summary",
        summary_name: str = "all_results",
    ) -> None:
        """
        Args:
        - layer_folder: str = "results", input layer wise result folder
        - summary_folder: str = "results/summary", output summary folder
        - summary_name: str = "all_results", name of output summary csv
        """

        self._layer_folder = os.path.normpath(layer_folder)
        self._summary_folder = checkNgen_folder(os.path.normpath(summary_folder))
        self._summary_name = summary_name

        self._full_results_df = self._summary_layer()

        # save master df
        print(f"Saving {self.summary_csv_path}...")
        self._full_results_df.to_csv(self.summary_csv_path, index=False)

    def _summary_layer(
        self,
    ):
        """
        A function for summary layer wise results

        ptp = pre-train percent
        """

        # init dataframe
        master_results = pd.DataFrame(
            columns=[
                "arch",
                "task",
                "model",
                "ablation",
                "ptp",
                "max_seq_num",
                "embseed",
                "sampleseed",
                "metric",
                "value",
            ]
        )

        # make value np array compatible
        master_results["value"] = master_results["value"].astype("object")

        for arch, ds_model in tqdm(
            itertools.product(PRETRAIN_ARCH_LIST, DS_MODEL_LIST),
            total=len(PRETRAIN_ARCH_LIST) * len(DS_MODEL_LIST),
            desc="ResultReorg total progress",
            position=0,
        ):

            print(f"Analyzing pretrain {arch} with downstream {ds_model}...")

            # get layerloss for normal embeds
            if arch == "carp":
                arch_info = CARP_MODEL_INFO
            elif arch == "esm":
                arch_info = ESM_MODEL_INFO
            if arch == "protxlstm":
                arch_info = PROTXLSTM_MODEL_INFO
            if arch == "protmamba":
                arch_info = PROTMAMBA_MODEL_INFO
            if arch == "poet":
                arch_info = POET_MODEL_INFO

            results_dict = defaultdict(dict)
            parser = ResultParser("onehot")
            onehot_path = os.path.join(self._layer_folder, f"{ds_model}-onehot")
            results_dict["onehot"] = parser.parse_folder(
                input_path=onehot_path,
                # output_path=input_path + "_layer",
            )
            master_results = self.extend_results_df(
                master_results, results_dict, ablation="onehot", ptp=0.0, arch=arch
            )

            for encoder_name in tqdm(
                arch_info.models.keys(),
                desc=f"Processing {arch}-{ds_model}",
                position=1,
                leave=True,
            ):
                parser = ResultParser(encoder_name)
                results_dict = defaultdict(dict)
                for ablation in DEFAULT_AB_LIST:
                    if ablation == "emb":
                        # pass
                        # # do normal embedding
                        ptp = 1.0
                        input_path = os.path.join(
                            self._layer_folder, f"{ds_model}-{arch}"
                        )
                        results_dict[encoder_name] = parser.parse_folder(
                            input_path=input_path,
                        )

                        # save to master
                        master_results = self.extend_results_df(
                            master_results, results_dict, ablation, ptp
                        )

                        if arch_info.models[encoder_name].checkpoint_names is not None:
                            for ckpt in arch_info.models[
                                encoder_name
                            ].checkpoint_names.keys():
                                results_dict[encoder_name] = defaultdict(dict)
                                ptp = float(ckpt)
                                input_path = os.path.join(
                                    self._layer_folder, f"{ds_model}-{arch}-{ckpt}"
                                )
                                results_dict[encoder_name] = parser.parse_folder(
                                    input_path=input_path,
                                )
                                master_results = self.extend_results_df(
                                    master_results, results_dict, ablation, ptp
                                )

                        for seq_num in [0, 50, 100]:
                            results_dict[encoder_name] = defaultdict(dict)
                            ptp = 1.0
                            input_path = os.path.join(
                                self._layer_folder,
                                f"{ds_model}-{arch}-maxseq_{seq_num}",
                            )
                            results_dict[encoder_name] = parser.parse_folder(
                                input_path=input_path,
                            )
                            master_results = self.extend_results_df(
                                master_results, results_dict, ablation, ptp, seq_num
                            )
                    elif ablation == "onehot":
                        pass

                    else:
                        # do ablations
                        ptp = 0.0
                        input_path = os.path.join(
                            self._layer_folder, f"{ds_model}-{arch}-{ablation}"
                        )
                        results_dict[encoder_name] = parser.parse_folder(
                            input_path=input_path,
                        )

                        # save to master
                        master_results = self.extend_results_df(
                            master_results, results_dict, ablation, ptp
                        )

        return master_results

    def extend_results_df(
        self, results_df, results_dict, ablation, ptp, seq_num=200, arch=None
    ):
        for model in results_dict.keys():
            if model != "onehot":
                arch = model.split("_")[0]
            if "esm" in arch:
                arch = "esm"
            for task in results_dict[model].keys():
                for seed in results_dict[model][task].keys():
                    for sampling_seed in results_dict[model][task][seed].keys():
                        for metric in results_dict[model][task][seed][
                            sampling_seed
                        ].keys():
                            # update metric and task if ss3
                            rename_metric = metric_simplifier(metric)
                            test_name = metric.split("_")[0]

                            if test_name in STRUCT_TESTS or test_name in RH_TESTS:
                                rename_metric = rename_metric.replace(test_name, "test")

                                # before update task: structure_ss3_tape_processed_noflatten
                                if "tape_processed" in task:
                                    rename_task = task.replace(
                                        "tape_processed", test_name
                                    )
                                else:
                                    split_list = task.split("_")
                                    rename_task = "_".join(
                                        split_list[:-1] + [test_name] + split_list[-1:]
                                    )
                            else:
                                rename_task = task

                            results_df = pd.concat(
                                [
                                    results_df,
                                    pd.DataFrame(
                                        {
                                            "arch": arch,
                                            "task": rename_task,
                                            "model": model,
                                            "ablation": ablation,
                                            "ptp": ptp,
                                            "max_seq_num": seq_num,
                                            "embseed": seed,
                                            "sampleseed": (
                                                sampling_seed
                                                if sampling_seed is not None
                                                else "sampling_seed-42"
                                            ),
                                            "metric": rename_metric,
                                            "value": [
                                                list(
                                                    results_dict[model][task][seed][
                                                        sampling_seed
                                                    ][metric]
                                                )
                                            ],
                                        }
                                    ),
                                ],
                                ignore_index=True,
                            )
        return results_df

    @property
    def summary_df(self) -> pd.DataFrame:
        """Return appended summary results"""
        return self._full_results_df

    @property
    def summary_csv_path(self) -> str:
        """Return summary csv path"""

        summary_csv_path = os.path.join(
            self._summary_folder, self._summary_name + ".csv"
        )

        if os.path.exists(summary_csv_path):
            print(f"Delete existing {summary_csv_path}...")
            os.remove(summary_csv_path)

        return summary_csv_path
