# Original code from Protein-Transfer under MIT License.

# Copyright (c) Microsoft Corporation.

# Modified by Antonia Ebner, 2026:
#   - adapted to allow for multiple sampling seeds
#   - removed plot_collage

"""Analyzing per layer output"""

from __future__ import annotations

from collections import defaultdict

import os
from glob import glob
import numpy as np

from src.encoding.encoding_classes import get_emb_info
from src.params.sys import RAND_SEED
from src.analysis.utils import METRIC_DICT
from src.utils import pickle_load, get_filename


class ResultParser:
    """A class for handling layer analysis"""

    def __init__(
        self,
        encoder_name,
        metric_dict: dict[list[str]] = METRIC_DICT,
    ):
        self.encoder_name = encoder_name
        self._metric_dict = metric_dict

    def parse_folder(
        self,
        input_path: str = "results/sklearn",
        # output_path: str = "results/sklearn_layer",
    ):
        """
        Args:
        - add_checkpoint: bool = True, if add checkpoint for carp
        - checkpoint_list: list = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        - input_path: str = "results/sklearn",
        - output_path: str = "results/sklearn_layer"
        - metric_dict: list[str] = ["train_mse", "test_ndcg", "test_rho"]
        """
        # get rid of the last "/" if any
        _input_path = os.path.normpath(input_path)
        # get the list of subfolders for each dataset
        _seed_folder = f"seed-{RAND_SEED}"
        _sampling_seed_folder = f"sampling_seed-{RAND_SEED}"
        _dataset_folders = glob(
            f"{_input_path}/{_seed_folder}/{_sampling_seed_folder}/*/*/*/{self.encoder_name}/*"
        ) 
        use_sample_seed = True

        if len(_dataset_folders) < 1:  # doesn't use sampling seeds
            _dataset_folders = glob(
                f"{_input_path}/{_seed_folder}/*/*/*/{self.encoder_name}/*"
            )
            use_sample_seed = False

        
        _analysis_dict = defaultdict(dict)

        # init a dict for metric params
        _metric_numb = defaultdict(dict)

        for (
            dataset_folder
        ) in (
            _dataset_folders
        ):  
            task_subfolder = dataset_folder.split(
                os.path.join(_input_path, _seed_folder) + "/"
            )[-1]
            if use_sample_seed:
                sample_seed_str, task, dataset, split, encoder_name, flatten_emb = (
                    task_subfolder.split("/")
                )
            else:
                task, dataset, split, encoder_name, flatten_emb = task_subfolder.split(
                    "/"
                )

            # get collage_name
            collage_name = f"{task}_{dataset}_{split}_{flatten_emb}"

            # get number of metircs
            _metric_numb[collage_name] = len(self._metric_dict[task])

            # parse results for plotting the collage and onehot

            _analysis_dict[collage_name] = defaultdict(dict)

            seed_list = glob(f"{_input_path}/*")

            for seed_folder in seed_list:
                seed_str = seed_folder.split("/")[-1]
                _analysis_dict[collage_name][seed_str] = defaultdict(dict)

                sample_seed_list = glob(f"{seed_folder}/*")

                if "sampling_seed" in sample_seed_list[0]:

                    for sample_seed_folder in sample_seed_list:

                        pkl_folder = dataset_folder.replace(
                            os.path.join(
                                _input_path, _seed_folder, _sampling_seed_folder
                            ),
                            sample_seed_folder,
                        )
                        sample_seed_str = sample_seed_folder.split("/")[-1]

                        if (
                            os.path.exists(pkl_folder)
                            and len(glob(f"{pkl_folder}/*.pkl")) > 0
                        ):
                            print(f"pkl_folder: {pkl_folder} exists. Processing...")
                            _analysis_dict[collage_name][seed_str][sample_seed_str] = (
                                self.parse_result_dicts(
                                    pkl_folder,
                                    task,
                                    dataset,
                                    split,
                                    encoder_name,
                                    flatten_emb,
                                )
                            )
                        else:
                            print(
                                f"pkl_folder: {pkl_folder} does not exist. Skipping..."
                            )
                else:
                    _analysis_dict[collage_name][seed_str]["none"] = (
                        self.parse_result_dicts(
                            dataset_folder,
                            task,
                            dataset,
                            split,
                            encoder_name,
                            flatten_emb,
                        )
                    )

        return _analysis_dict


    def parse_result_dicts(
        self,
        folder_path: str,
        task: str,
        dataset: str,
        split: str,
        encoder_name: str,
        flatten_emb: bool | str,
    ):
        """
        Parse the output result dictionaries for plotting

        Args:
        - folder_path: str, the folder path for the datasets
        - task: str, the task name
        - dataset: str, the dataset name
        - split: str, the split name
        - encoder_name: str, the encoder name
        - flatten_emb: bool | str, if the embedding is flatten

        Returns:
        - dict, encode name as key with a dict as its value
            where metric name as keys and the array of losses as values
        - str, details for collage plot
        """

        # get the list of output pickle files
        pkl_list = glob(f"{folder_path}/*.pkl")

        # should be results/sklearn-carp-stat/seed-42/proeng/thermo/mixed_split/carp_600k/mean/carp_600k-mean-layer_9.pkl

        _, _, max_layer_numb = get_emb_info(encoder_name)

        # init the ouput dict
        output_numb_dict = {
            metric: np.zeros([max_layer_numb]) for metric in self._metric_dict[task]
        }

        print(pkl_list)

        # loop through the list of the pickle files (all layers)
        for pkl_file in pkl_list:
            # get the layer number
            layer_numb = int(get_filename(pkl_file).split("-")[-1].split("_")[-1])
            # load the result dictionary
            try:
                result_dict = pickle_load(pkl_file)
            except Exception as e:
                print(f"{pkl_file} with err: ", e)

            # populate the processed dictionary
            for metric in self._metric_dict[task]:
                subset, kind = metric.split("_")
                subset = subset.replace("-", "_")
                kind = kind.replace("-", "_") if "top5" in kind else kind
                if kind == "rho":
                    output_numb_dict[metric][layer_numb] = result_dict[subset][kind][0]
                else:
                    output_numb_dict[metric][layer_numb] = result_dict[subset][kind]

        return output_numb_dict
