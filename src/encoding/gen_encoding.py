# Original code from Protein-Transfer under MIT License.

# Copyright (c) Microsoft Corporation. 

# Modified by Antonia Ebner, 2026:
#   - extended for protxlstm, protmamba and poet
# 
from __future__ import annotations

import os
import sys
import tables

import numpy as np

from src.utils import get_folder_file_names, checkNgen_folder
from src.params.emb import MAX_SEQ_LEN
from src.params.sys import RAND_SEED
from src.encoding.encoding_classes import (
    get_emb_info,
    OnehotEncoder,
    CARPEncoder,
    ProtxLSTMEncoder,
    ProtMambaEncoder,
    PoETEncoder,
)
from src.preprocess.data_process import ProtranDataset


class GenerateEmbeddings:
    """A class for generating and saving embeddings"""

    def __init__(
        self,
        dataset_path: str,
        encoder_name: str,
        checkpoint: float = 1,
        checkpoint_folder: str = "pretrain_checkpoints/",
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
        embed_batch_size: int = 128,
        flatten_emb: bool | str = False,
        seq_start_idx: bool | int = False,
        seq_end_idx: bool | int = False,
        subset_list: list[str] = ["train", "val", "test"],
        embed_folder: str = "embeddings",
        max_context_length: int = 200_000,
        max_context_sequences: int = 200,
        max_similarity: float = 0.98,
        max_dissimilarity: float = 0.7,
        yield_with_context: bool = True,
        is_mutation: bool = False,
        chunk_chunk_size: int = 2**14,
        mlstm_chunksize: int = 1024,
        sampling_seed: int = RAND_SEED,
        msa_path: str = None,
        manual_layer_min: bool | int = False,
        manual_layer_max: bool | int = False,
        max_seq_len: int = -1,
        overwrite_file: bool = False,
        sort_context: bool = False,
        **encoder_params,
    ) -> None:
        """
        Args:
        - dataset_path: str, full path to the dataset, in pkl or panda readable format
            columns include: sequence, target, set, validation,
            mut_name (optional), mut_numb (optional)
        - encoder_name: str, the name of the encoder
        - checkpoint: float = 1, the 0.5, 0.25, 0.125 checkpoint of the CARP encoder or full
        - checkpoint_folder: str = "pretrain_checkpoints/carp", folder for carp encoders
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        - embed_batch_size: int, set to 0 to encode all in a single batch
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - seq_start_idx: bool | int = False, the index for the start of the sequence
        - seq_end_idx: bool | int = False, the index for the end of the sequence
        - subset_list: list of str, train, val, test, or ss3 tasks including 'cb513', 'ts115', 'casp12'
        - embed_folder: str = "embeddings", the parent folder for embeddings
        - encoder_params: kwarg, additional parameters for encoding
        """

        assert (
            yield_with_context or is_mutation
        ), "yield_with_context and is_mutation cannot both be false"

        self.dataset_path = dataset_path
        self.encoder_name = encoder_name
        self.reset_param = reset_param
        self.resample_param = resample_param
        self.flatten_emb = flatten_emb

        self.embed_folder = embed_folder

        # append emb info
        if checkpoint != 1:
            self.embed_folder = f"{self.embed_folder}-{str(checkpoint)}"

        # append init info
        if self.reset_param and "-rand" not in self.embed_folder:
            self.embed_folder = f"{self.embed_folder}-rand"
            if self.reset_param in ["intra_seq", "inter_seq"]:
                self.embed_folder = f"{self.embed_folder}-{self.reset_param}"

        if self.resample_param and "-stat" not in self.embed_folder:
            self.embed_folder = f"{self.embed_folder}-stat"

        # append seed info
        self.embed_folder = checkNgen_folder(
            os.path.join(self.embed_folder, f"seed-{str(embed_torch_seed)}")
        )

        self.encoder_name, encoder_class, total_emb_layer = get_emb_info(
            self.encoder_name
        )

        # assert encoder_class != OnehotEncoder, "Generate onehot on the fly instead"
        # add in the max_seq_len for Onehot
        if encoder_class == OnehotEncoder and self.flatten_emb != False:
            encoder_params["max_seq_len"] = MAX_SEQ_LEN
            embed_rescale = MAX_SEQ_LEN
        else:
            embed_rescale = 1

        if encoder_class in [CARPEncoder, ProtxLSTMEncoder]:
            encoder_params["checkpoint"] = checkpoint
            encoder_params["checkpoint_folder"] = checkpoint_folder

        if encoder_class in [ProtxLSTMEncoder, ProtMambaEncoder, PoETEncoder]:
            from src.preprocess.dataloaders import DownstreamMemmapDataset

            if encoder_class in [ProtxLSTMEncoder, ProtMambaEncoder]:
                encoder_params["chunk_chunk_size"] = chunk_chunk_size
                encoder_params["mlstm_chunksize"] = mlstm_chunksize
            self.embed_folder = checkNgen_folder(
                os.path.join(self.embed_folder, f"sampling_seed-{str(sampling_seed)}")
            )
            if encoder_class in [ProtxLSTMEncoder, PoETEncoder]:
                assert embed_batch_size == 1, "embedding batch size must be 1!"

        # get the encoder
        self._encoder = encoder_class(
            encoder_name=encoder_name,
            reset_param=reset_param,
            resample_param=resample_param,
            embed_torch_seed=embed_torch_seed,
            **encoder_params,
        )

        if self.flatten_emb == False:
            flatten_emb_name = "noflatten"
        else:
            flatten_emb_name = self.flatten_emb

        # get the folder name
        dataset_folder, _ = get_folder_file_names(
            parent_folder=self.embed_folder,
            dataset_path=self.dataset_path,
            encoder_name=self.encoder_name,
            embed_layer=0,
            flatten_emb=flatten_emb_name,
        )

        # Close all the open files
        tables.file._open_files.close_all()

        manual_layer_min = int(manual_layer_min) if manual_layer_min else 0
        manual_layer_max = (
            int(manual_layer_max) if manual_layer_max else total_emb_layer
        )

        for subset in subset_list:

            print(f"Generating embedding for {subset}...")

            # get the dataset to be encoded
            if encoder_class in [ProtxLSTMEncoder, ProtMambaEncoder, PoETEncoder]:
                assert msa_path is not None, "No MSA memmap file path passed!"
                dataset_path_split = (
                    os.path.splitext(dataset_path)[0].strip("/").split("/")
                )
                if dataset_path_split[-2] == "gb1":
                    dataset_path_split[-2] = "gb1_trunc"

                dset_path = (
                    os.path.join(*dataset_path_split) + "_trunc.csv"
                    if dataset_path_split[-2] in ["gb1", "gb1_trunc"]
                    else os.path.join(*dataset_path_split) + ".csv"
                )
                subset_path = "/".join(dataset_path_split) + f"_{subset}_ids.csv"

                ds = DownstreamMemmapDataset(
                    dataset_path=dset_path,
                    msa_memmap_path=msa_path + ".dat",
                    msa_memmap_meta_path=msa_path + "_indices.csv",
                    subset_path=subset_path,
                    msa_sim_memmap_path=msa_path + "_similarity.dat",
                    msa_weights_memmap_path=msa_path + "_weights.dat",
                    msa_sim_meta_memmap_path=msa_path + "_meta_indices.csv",
                    max_context_length=max_context_length,
                    max_context_sequences=max_context_sequences,
                    max_similarity=max_similarity,
                    max_dissimilarity=max_dissimilarity,
                    is_mutation=is_mutation,
                    yield_with_context=yield_with_context,
                    seed=sampling_seed,
                    max_seq_len=max_seq_len,
                    sort_context=sort_context,
                )

                # precompute context
                if is_mutation and not yield_with_context:
                    context = ds.get_context(ds.wildtype)
                    self._encoder.state = self._encoder.precompute_context_state(
                        context, chunk_chunk_size=chunk_chunk_size
                    )
                    if encoder_class in [PoETEncoder]:
                        # poet_maxlen = ds._max_seq_len + 1
                        self._encoder.state = (
                            self._encoder.model.logits_allocate_memory(
                                memory=self._encoder.state,
                                batch_size=embed_batch_size,
                                length=ds._max_seq_len + 1,  # add start & stop token
                            )
                        )
            else:
                ds = ProtranDataset(
                    dataset_path=dataset_path,
                    subset=subset,
                    encoder_name=encoder_name,
                    reset_param=reset_param,
                    resample_param=resample_param,
                    embed_torch_seed=embed_torch_seed,
                    embed_batch_size=embed_batch_size,
                    flatten_emb=flatten_emb,
                    embed_folder=None,
                    embed_layer="all",
                    seq_start_idx=seq_start_idx,
                    seq_end_idx=seq_end_idx,
                    if_encode_all=False,
                    **encoder_params,
                )

            # get the max seq len from the dataset to pad the embeddings
            self._max_seq_len = ds._max_seq_len

            if isinstance(self._encoder, ProtMambaEncoder):
                self._encoder.max_sequence_length = self._max_seq_len
            elif isinstance(self._encoder, PoETEncoder):
                self._encoder.update_max_sequence_length(ds._max_seq_len + 1)

            # get the dim of the array to be saved
            # without flattening
            if self.flatten_emb == False:
                earray_dim = (0, self._max_seq_len, self._encoder.embed_dim)
            else:
                earray_dim = (0, self._encoder.embed_dim * embed_rescale)

            init_array_list = [None] * total_emb_layer


            file_path = os.path.join(
                checkNgen_folder(os.path.join(dataset_folder, subset)),
                "embedding.h5",
            )

            # check all the embedding file h5 files
            # to remove old ones before generating new ones
            if os.path.isfile(file_path) and overwrite_file:
                print("Overwritting {0}".format(file_path))
                os.remove(file_path)

            elif os.path.isfile(file_path):
                print("File {0} already exits\nQuitting".format(file_path))
                sys.exit()

            f = tables.open_file(file_path, mode="a")
            # init file open
            for emb_layer in range(total_emb_layer):
                if manual_layer_min <= emb_layer <= manual_layer_max:
                    init_array_list[emb_layer] = f.create_earray(
                        f.root,
                        "layer" + str(emb_layer),
                        tables.Float32Atom(),
                        earray_dim,
                    )

            # use the encoder generator for batch emb
            # assume no labels included
            for i, encoded_batch_dict in enumerate(
                self._encoder.encode(
                    mut_seqs=list(range(len(ds))),
                    dataset=ds,
                    batch_size=embed_batch_size,
                    flatten_emb=flatten_emb,
                )
            ):
                for emb_layer, emb in encoded_batch_dict.items():
                    if manual_layer_min <= emb_layer <= manual_layer_max:
                        # if no flattening, pad all embedding to the max_seq_len
                        if flatten_emb == False:
                            padded_emb = np.pad(
                                emb,
                                pad_width=(
                                    (0, 0),
                                    (0, self._max_seq_len - emb.shape[1]),
                                    (0, 0),
                                ),
                            )
                        else:
                            padded_emb = emb

                        getattr(f.root, "layer" + str(emb_layer)).append(padded_emb)

            f.close()
