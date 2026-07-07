# Original code from Protein-Transfer under MIT License.

# Copyright (c) Microsoft Corporation. 

# Modified by Antonia Ebner, 2026:
#   - extended for protxlstm, protmamba and poet
#   - refactored the code
# 
"""Encoding classes.

Hierarchy
---------
``AbstractEncoder`` (ABC)
├── ``OnehotEncoder``
├── ``ESMEncoder``
├── ``CARPEncoder``
└── ``ContextEncoder`` (ABC, query/context MSA models)
    ├── ``ProtxLSTMEncoder``
    ├── ``ProtMambaEncoder``
    └── ``PoETEncoder``

The parent classes own everything generic (batching, flattening, pooling,
parameter logging, and the reset/resample orchestration).  Every model family
implements only the pieces that differ: how its weights are loaded, how its
weights are re-initialised/shuffled for ablations (``_reset_weights`` /
``_resample_weights``), and its forward pass (``_encode_batch``).

NOTE ON REPRODUCIBILITY
-----------------------
This refactor is purely structural: code blocks that consume the torch RNG were
moved verbatim, keeping the same ``seed_all`` timing and the same iteration
order over ``state_dict()``.  Several pre-existing quirks are intentionally kept
as-is so the produced embeddings stay byte-identical (flagged with
``# NOTE: kept as-is for reproducibility``).  Do not "fix" them here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection
from collections.abc import Iterable, Sequence
from collections import OrderedDict

import os
import math
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import Parameter
from torch.nn.init import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    uniform_,
    normal_,
    constant_,
    ones_,
    zeros_,
    _calculate_fan_in_and_fan_out,
)
import torch.nn.functional as F
from torch.utils.data import Dataset
from sequence_models.pretrained import load_model_and_alphabet

from omegaconf import OmegaConf


from src.params.aa import AA_NUMB, AA_TO_IND
from src.params.emb import (
    PROTXLSTM_INFO,
    TRANSFORMER_INFO,
    CARP_INFO,
    CARP_CHECKPOINTS,
    PROTXLSTM_CHECKPOINTS,
    PROTMAMBA_INFO,
    POET_INFO,
)
from src.params.sys import DEVICE, RAND_SEED
from src.utils import seed_all

# --------------------------------------------------------------------------- #
# Optional model-backend imports.                                             #
#                                                                             #
# The autoregressive protein-LM backends (ProtxLSTM, ProtMamba, PoET) and the #
# shared dataloaders pull in heavy, environment-specific dependencies (custom #
# CUDA kernels, flash-attention, ...) that are not installable everywhere.    #
# They are imported eagerly but defensively so that *importing this module    #
# never fails*: when a backend is unavailable its symbols stay ``None`` and   #
# ``_require`` raises a clear error only when that encoder is instantiated.    #
# --------------------------------------------------------------------------- #

try:
    from src.preprocess.dataloaders import (
        ProteinDataCollator,
        DownstreamMemmapDataset,
    )
except Exception as err:  # noqa: BLE001 - optional backend dependency
    ProteinDataCollator = DownstreamMemmapDataset = None
    _DATALOADERS_ERR = err
else:
    _DATALOADERS_ERR = None

# ``load_model`` is the generic loader (takes a ``model_class``); it is shared by
# the ProtxLSTM and ProtMamba encoders. PoET ships its own loader, aliased below.
try:
    from src.models.protxlstm.utils import load_model
    from src.models.protxlstm.xlstm_head import xLSTMLMHeadModel, xLSTMConfig
except Exception as err:  # noqa: BLE001 - optional backend dependency
    load_model = xLSTMLMHeadModel = xLSTMConfig = None
    _PROTXLSTM_ERR = err
else:
    _PROTXLSTM_ERR = None

try:
    from src.models.mamba.mamba import MambaLMHeadModelwithPosids, MambaConfig
    from src.models.mamba.utils_generation import InferenceParams
except Exception as err:  # noqa: BLE001 - optional backend dependency
    MambaLMHeadModelwithPosids = MambaConfig = InferenceParams = None
    _PROTMAMBA_ERR = err
else:
    _PROTMAMBA_ERR = None

try:
    from src.models.poet.utils import load_model as load_poet_model
    from src.models.poet.poet import PoET
    from src.models.poet.alphabets import Uniprot21
except Exception as err:  # noqa: BLE001 - optional backend dependency
    load_poet_model = PoET = Uniprot21 = None
    _POET_ERR = err
else:
    _POET_ERR = None


def _require(symbol, backend: str, error: Exception | None):
    """Raise a clear error if an optional backend failed to import."""
    if symbol is None:
        raise ImportError(
            f"The '{backend}' backend is unavailable in this environment; "
            f"install its dependencies to use this encoder. "
            f"Original import error: {error!r}"
        ) from error



def cal_bound(model: torch.nn.Module, layer_name: str):
    """Return bound for reinit given model and layer name"""
    assert "bias" in layer_name, f"no bias in {layer_name}"
    fan_in, _ = _calculate_fan_in_and_fan_out(
        model.state_dict()[layer_name.replace("bias", "weight")]
    )
    return 1 / math.sqrt(fan_in) if fan_in > 0 else 0


class AbstractEncoder(ABC):
    """
    An abstract encoder class to fill in for different kinds of encoders

    All encoders will have an "encode" function
    """

    def __init__(
        self,
        encoder_name: str = "",
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
    ):
        """
        Args:
        - encoder_name: str, the name of the encoder, default empty for onehot
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """

        self._encoder_name = encoder_name

        assert reset_param * resample_param != 1, "Choose reset OR resample param"

        self._reset_param = reset_param
        self._resample_param = resample_param
        self._embed_torch_seed = embed_torch_seed

    # ------------------------------------------------------------------ #
    # Parameter-sum logging (diagnostic only, no effect on the RNG)       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _param_sum(model: torch.nn.Module) -> float:
        """Sum of all parameters, cast to float32 so bf16 models work too."""
        s = 0
        for p in model.parameters():
            s += np.sum(p.cpu().float().data.numpy())
        return s

    def _log_param_sum(self, model: torch.nn.Module, stage: str):
        """Print the running parameter sum used to sanity-check reproducibility."""
        print(
            f"[{self.__class__.__name__} | {self._encoder_name} | "
            f"{self.emb_ablation}] param sum {stage}: {self._param_sum(model)}"
        )

    # ------------------------------------------------------------------ #
    # Ablation orchestration                                              #
    # ------------------------------------------------------------------ #
    def _reset_weights(self, model: torch.nn.Module):
        """Re-initialise weights (``rand`` ablation). Override where supported."""
        pass

    def _resample_weights(self, model: torch.nn.Module):
        """Shuffle weights in place (``stat`` ablation). Override where supported."""
        pass

    def reset_resample_param(self, model: torch.nn.Module):
        """
        Initiate parameters in the PyTorch model. Following:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer

        Dispatches to the encoder-specific ``_reset_weights`` / ``_resample_weights``
        hook.  The ``seed_all`` call happens here, before any RNG is consumed, so
        every family shares the same seeding point.

        Args:
        - model: torch.nn.Module, the input model

        Returns:
        - torch.nn.Module, the (re-initialised) model
        """

        seed_all(self._embed_torch_seed)

        print(
            "Running {} ablation for {} with {} inside reset_resample_param...".format(
                self.emb_ablation, self._encoder_name, self._embed_torch_seed
            )
        )

        self._log_param_sum(model, "input to reset_resample_param")

        if self._reset_param:
            print(f"Reinit params for {self._encoder_name} ...")
            self._reset_weights(model)

        elif self._resample_param:
            print(f"Resample params for {self._encoder_name} ...")
            self._resample_weights(model)

        else:
            print("Not changing the model")

        self._log_param_sum(model, "after reset_resample_param")

        return model

    def encode(
        self,
        mut_seqs: Sequence[int],
        dataset: Dataset = None,
        batch_size: int = 0,
        flatten_emb: bool | str = False,
        mut_names: Sequence[str] | str | None = None,
    ) -> Iterable[np.ndarray]:
        """
        A function takes a list of sequences to yield a batch of encoded elements

        Args:
        - mut_seqs: list of str or str, mutant sequences of the same length
        - batch_size: int, set to 0 to encode all in a single batch
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - generator: dict with layer number as keys and
            encoded flattened sequence with or without labels as value
        """

        # if isinstance(mut_seqs, str):
        #     mut_seqs = [mut_seqs]

        # If the batch size is 0, then encode all at once in a single batch
        if batch_size == 0:
            yield self._encode_batch(
                mut_seqs=mut_seqs,
                dataset=dataset,
                flatten_emb=flatten_emb,
                mut_names=mut_names,
            )

        # Otherwise, yield chunks of encoded sequence
        else:

            for i in tqdm(range(0, len(mut_seqs), batch_size)):

                # figure out what mut_names to feed in
                if mut_names is None:
                    mut_name_batch = mut_names
                else:
                    mut_name_batch = mut_names[i : i + batch_size]

                yield self._encode_batch(
                    mut_seqs=mut_seqs[i : i + batch_size],
                    dataset=dataset,
                    flatten_emb=flatten_emb,
                    mut_names=mut_name_batch,
                )

    def _pooling_seq_len(self, mut_seq, encoded_mut_seq) -> int:
        """
        Sequence length (excluding special tokens) used for mean/max pooling.

        Base handling covers ESM ``(name, seq)`` tuples and CARP ``[seq]`` lists;
        context models override this for their query dicts.
        """
        # if the emb has label from esm
        if len(mut_seq) == 2:
            return len(mut_seq[1])  # variable sequence length!
        # if the emb is carp
        elif len(mut_seq) == 1:
            return len(mut_seq[0])
        else:
            return len(mut_seq)

    def flatten_encode(
        self,
        encoded_mut_seqs: np.ndarray,
        flatten_emb: bool | str,
        mut_seqs: Sequence[str] | str,
    ) -> np.ndarray:
        """
        Flatten the embedding or just return the encoded mutants.

        Args:
        - encoded_mut_seqs: np.ndarray, shape [batch_size, seq_len, embed_dim]
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
            - True -> shape [batch_size, seq_len * embed_dim]
            - "max" or "mean" -> shape [batch_size, embed_dim]
            - False or everything else -> [batch_size, seq_len, embed_dim]

        Returns:
        - np.ndarray, shape depends on flatten_emb parameter
        """

        assert (
            encoded_mut_seqs.shape[-1] == self._embed_dim
        ), f"encode last dim {encoded_mut_seqs.shape[-1]} != embed dim {self._embed_dim}"

        if flatten_emb in [True, "flatten", "flattened", ""]:
            # shape [batch_size, seq_len * embed_dim]
            return encoded_mut_seqs.reshape(encoded_mut_seqs.shape[0], -1)

        elif isinstance(flatten_emb, str):
            # init out put seq_reps should be in dim [batch_size, embed_dim]
            seq_reps = np.empty((encoded_mut_seqs.shape[0], self._embed_dim))
            for i, encoded_mut_seq in enumerate(encoded_mut_seqs):
                seq_len = self._pooling_seq_len(mut_seqs[i], encoded_mut_seq)

                assert seq_len not in [1, 2], "Check emb pooling len!"

                if flatten_emb == "mean":
                    # print("seq len before avg", len(encoded_mut_seq[:seq_len]))
                    seq_reps[i] = encoded_mut_seq[:seq_len].mean(0)
                elif flatten_emb == "max":
                    seq_reps[i] = encoded_mut_seq[:seq_len].max(0)

            return seq_reps

        else:
            # print("No embedding flattening")
            # [batch_size, seq_len, embed_dim]
            return encoded_mut_seqs

    @abstractmethod
    def _encode_batch(
        mut_seqs: Sequence[int] | int,
        dataset: Dataset,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encode a single batch of mut_seqs
        """
        pass

    @property
    def embed_dim(self) -> int:
        """The dim of the embedding"""
        return self._embed_dim

    @property
    def max_emb_layer(self) -> int:
        """The max layer nubmer of the embedding"""
        return self._max_emb_layer

    @property
    def encoder_name(self) -> str:
        """The name of the encoding method"""
        return self._encoder_name

    @property
    def emb_ablation(self) -> str:
        """The ablation of the encoding method"""
        if self._reset_param:
            return "rand"
        elif self._resample_param:
            return "stat"
        else:
            return "none"


class OnehotEncoder(AbstractEncoder):
    """
    Build a onehot encoder
    """

    def __init__(
        self,
        max_seq_len: int,
        encoder_name: str = "",
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of CARP_INFO
        - max_seq_len: int, the longest sequence length
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """
        super().__init__(encoder_name, reset_param, resample_param)

        self.max_seq_len = max_seq_len

        if encoder_name not in (
            TRANSFORMER_INFO.keys() and CARP_INFO.keys() and PROTXLSTM_INFO.keys()
        ):
            self._encoder_name = "onehot"
            self._embed_dim, self._max_emb_layer = AA_NUMB, 0
            self._include_input_layer = True

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

        if reset_param or resample_param:
            self._reset_param = False
            self._resample_param = False
            print(
                f"Onehot encoding reset or resample param not allowed. /n \
                    Setting both to {self._reset_param} ..."
            )

    def _encode_batch(
        self,
        mut_seqs: Sequence[int] | int,
        dataset: Dataset,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:

        if isinstance(mut_seqs, int):
            mut_seqs = [mut_seqs]

        encoded_mut_seqs = []
        mut_seqs = [dataset.sequence[m] for m in mut_seqs]

        for mut_seq in mut_seqs:
            # padding: (top, bottom), (left, right)
            encoded_mut_seqs.append(
                np.pad(
                    np.array(np.eye(AA_NUMB)[[AA_TO_IND[aa] for aa in mut_seq]]),
                    pad_width=((0, self.max_seq_len - len(mut_seq)), (0, 0)),
                )
            )

        return {
            0: self.flatten_encode(
                encoded_mut_seqs=np.array(encoded_mut_seqs).astype(np.float32),
                flatten_emb=flatten_emb,
                mut_seqs=mut_seqs,
            ).astype(np.float32)
        }


class ESMEncoder(AbstractEncoder):
    """
    Build an ESM encoder
    """

    def __init__(
        self,
        encoder_name: str,
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
        iftrimCLS: bool = True,
        iftrimEOS: bool = True,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of TRANSFORMER_INFO
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        - iftrimCLS: bool, whether to trim the first classifification token
        - iftrimEOS: bool, whether to trim the end of sequence token, if exists
        """

        super().__init__(encoder_name, reset_param, resample_param, embed_torch_seed)

        print(f"Seed for ESMEncoder: {self._embed_torch_seed}")

        self._iftrimCLS = iftrimCLS
        self._iftrimEOS = iftrimEOS

        # get transformer dim and layer info
        self._embed_dim, self._max_emb_layer, _ = TRANSFORMER_INFO[self._encoder_name]

        # esm has the input representation
        self._include_input_layer = True

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

        self.model, self.alphabet = torch.hub.load(
            "facebookresearch/esm:main", model=self._encoder_name
        )
        self.batch_converter = self.alphabet.get_batch_converter()

        self._log_param_sum(self.model, "after load, before ablation")

        # if reset or resample weights
        self.model = self.reset_resample_param(model=self.model)

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        self._log_param_sum(self.model, "after ablation")

        expected_num_layers = int(self._encoder_name.split("_")[-3][1:])
        assert (
            expected_num_layers == self._max_emb_layer
        ), "Wrong ESM model name or layer"

    def _reset_weights(self, model: torch.nn.Module):
        """
        Re-initialise ESM-1 / ESM-1b weights following the layer-wise scheme
        below.

        ESM1b:

        layers.n.self_attn.k_proj.weight: dim 2         [nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))]
        layers.n.self_attn.v_proj.weight: dim 2         [nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))]
        layers.n.self_attn.q_proj.weight: dim 2         [nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))]
        layers.n.self_attn.out_proj.weight: dim 2       [nn.init.xavier_uniform_(self.out_proj.weight)]

        layers.n.self_attn.k_proj.bias: dim 1           [nn.init.uniform_(self.bias, -bound, bound)]
        layers.n.self_attn.v_proj.bias: dim 1           [nn.init.uniform_(self.bias, -bound, bound)]
        layers.n.self_attn.q_proj.bias: dim 1           [nn.init.uniform_(self.bias, -bound, bound)]

        layers.n.self_attn.out_proj.bias: dim 1         [nn.init.constant_(self.out_proj.bias, 0.0)]

        layers.n.self_attn_layer_norm.weight: dim 1     [nn.Parameter(torch.ones(hidden_size))]
        layers.n.final_layer_norm.weight: dim 1         [nn.Parameter(torch.ones(hidden_size))]
        emb_layer_norm_before.weight: dim 1             [nn.Parameter(torch.ones(hidden_size))]
        emb_layer_norm_after.weight: dim 1              [nn.Parameter(torch.ones(hidden_size))]

        layers.n.self_attn_layer_norm.bias: dim 1       [nn.Parameter(torch.zeros(hidden_size))]
        layers.n.final_layer_norm.bias: dim 1           [nn.Parameter(torch.zeros(hidden_size))]
        emb_layer_norm_before.bias: dim 1               [nn.Parameter(torch.zeros(hidden_size))]
        emb_layer_norm_after.bias: dim 1                [nn.Parameter(torch.zeros(hidden_size))]

        layers.n.fc1.weight: dim 2                      [nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))]
        layers.n.fc2.weight: dim 2                      [nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))]
        contact_head.regression.weight: dim 2           [nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))]

        layers.n.fc1.bias: dim 1                        [nn.init.uniform_(self.bias, -bound, bound)]
        layers.n.fc2.bias: dim 1                        [nn.init.uniform_(self.bias, -bound, bound)]
        contact_head.regression.bias: dim 1             [nn.init.uniform_(self.bias, -bound, bound)]

        embed_positions.weight: dim 2                   [nn.init.normal_(self.weight)]
        """
        print(f"Updating esm {self._encoder_name} weights...")
        for layer_name, p in model.state_dict().items():
            # what esm1b and esm1 have in common
            if "_proj" in layer_name:
                if "weight" in layer_name:
                    if "out" in layer_name:
                        xavier_uniform_(p)
                    else:
                        xavier_uniform_(p, gain=1 / math.sqrt(2))
                elif "bias" in layer_name:
                    if "out" in layer_name:
                        constant_(p, 0.0)
                    else:
                        bound = cal_bound(model=model, layer_name=layer_name)
                        uniform_(p, -bound, bound)

            # esm1b enced up using LayerNorm so the same
            if "layer_norm" in layer_name:
                if "weight" in layer_name:
                    Parameter(torch.ones_like(p))
                elif "bias" in layer_name:
                    Parameter(torch.zeros_like(p))

            if ("layers" and "fc" in layer_name) or (
                "contact_head" in layer_name
            ):
                if "weight" in layer_name:
                    kaiming_uniform_(p, a=math.sqrt(5))
                elif "bias" in layer_name:
                    bound = cal_bound(model=model, layer_name=layer_name)
                    uniform_(p, -bound, bound)

            if "esm1b_" in self._encoder_name:

                if "embed_positions" in layer_name:
                    normal_(p)

                if layer_name == "lm_head.weight":
                    xavier_uniform_(p)

                if layer_name == "lm_head.bias" or "lm_head.layer_norm.bias":
                    Parameter(torch.zeros_like(p))

                if "dense" in layer_name:
                    if "weight" in layer_name:
                        kaiming_uniform_(p, a=math.sqrt(5))
                    elif "bias" in layer_name:
                        bound = cal_bound(model=model, layer_name=layer_name)
                        uniform_(p, -bound, bound)

            elif "esm1_" and "bias_" in self._encoder_name:
                xavier_normal_(p)

    def _resample_weights(self, model: torch.nn.Module):
        print(f"Updating esm {self._encoder_name} weights...")
        resample_state = model.state_dict()
        for layer_name, p in model.state_dict().items():
            if (
                ("embed_tokens" not in layer_name)
                and ("embed_out" not in layer_name)
                and ("_float_tensor" not in layer_name)
            ):
                # shuffle all dim
                resample_state[layer_name] = p.view(-1)[
                    torch.randperm(p.view(-1).shape[0])
                ].view(p.shape)
        model.load_state_dict(resample_state)

    def _encode_batch(
        self,
        mut_seqs: Sequence[int] | int,
        dataset: Dataset,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encodes a batch of mutant sequences.

        Args:
        - mut_seqs: list of str or str, mutant sequences of the same length
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - np.ndarray or a tuple(np.ndarray, list[str]) where the list is batch_labels
        """

        if isinstance(mut_names, str):
            mut_names = [mut_names]
        if isinstance(mut_seqs, int):
            mut_seqs = [mut_seqs]

        # pair the mut_names and mut_seqs
        if mut_names is not None:
            assert len(mut_names) == len(
                mut_seqs
            ), "mutant_name and mut_seqs different length"
            mut_seqs = [(n, dataset.sequence[m]) for (n, m) in zip(mut_names, mut_seqs)]
        else:
            mut_seqs = [("", dataset.sequence[m]) for m in mut_seqs]

        # convert raw mutant sequences to tokens
        batch_labels, _, batch_tokens = self.batch_converter(mut_seqs)
        batch_tokens = batch_tokens.to(DEVICE)

        # Turn off gradients and pass the batch through
        with torch.no_grad():
            # shape [batch_size, seq_len + pad, embed_dim]

            dict_encoded_mut_seqs = self.model(
                batch_tokens, repr_layers=list(range(self._max_emb_layer + 1))
            )["representations"]

        for layer, encoded_mut_seqs in dict_encoded_mut_seqs.items():

            encoded_mut_seqs = encoded_mut_seqs.cpu().numpy()
            # https://github.com/facebookresearch/esm/blob/main/esm/data.py
            # from_architecture

            # trim off initial classification token [CLS]
            # both "ESM-1" and "ESM-1b" have prepend_bos = True
            if self._iftrimCLS and self._encoder_name.split("_")[0] in [
                "esm1",
                "esm1b",
            ]:
                encoded_mut_seqs = encoded_mut_seqs[:, 1:, :]

            # trim off end-of-sequence token [EOS]
            # only "ESM-1b" has append_eos = True
            if self._iftrimEOS and self._encoder_name.split("_")[0] == "esm1b":
                encoded_mut_seqs = encoded_mut_seqs[:, :-1, :]

            if mut_names is not None:
                dict_encoded_mut_seqs[layer] = (
                    self.flatten_encode(
                        encoded_mut_seqs=encoded_mut_seqs,
                        flatten_emb=flatten_emb,
                        mut_seqs=mut_seqs,
                    ),
                    batch_labels,
                )
            else:
                dict_encoded_mut_seqs[layer] = self.flatten_encode(
                    encoded_mut_seqs=encoded_mut_seqs,
                    flatten_emb=flatten_emb,
                    mut_seqs=mut_seqs,
                )

        return dict_encoded_mut_seqs


class CARPEncoder(AbstractEncoder):
    """
    Build a CARP encoder
    """

    def __init__(
        self,
        encoder_name: str,
        checkpoint: float = 1,
        checkpoint_folder: str = "pretrain_checkpoints/carp",
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of CARP_INFO
        - checkpoint: float = 1, the 0.5, 0.25, 0.125 checkpoint of the CARP encoder or full
        - checkpoint_folder: str = "pretrain_checkpoints/carp", folder for carp encoders
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """

        super().__init__(encoder_name, reset_param, resample_param, embed_torch_seed)

        print(f"Seed for CARPEncoder: {self._embed_torch_seed}")

        self.model, self.collater = load_model_and_alphabet(self._encoder_name)

        self._log_param_sum(self.model, "after load, before checkpoint")

        # load checkpoint unless default to full
        if checkpoint != 1:

            # get the checkpoint number from the CARP_CHECKPOINTS dict
            # ie {"carp_600k": {"1/2": 239263, ...}, ...}
            # to get 'pretrain_checkpoints/carp/carp_600k/checkpoint239263.tar'

            checkpoint_path = (
                f"{os.path.normpath(checkpoint_folder)}/{encoder_name}/"
                f"checkpoint{str(CARP_CHECKPOINTS[encoder_name][checkpoint])}.tar"
            )

            print(
                f"Loading {encoder_name} {checkpoint} checkpoint from {checkpoint_path}..."
            )

            # get the dict with dict_keys(['model_state_dict', ...])
            checkpoint_dict = torch.load(checkpoint_path, map_location=DEVICE)

            self.model.load_state_dict(
                OrderedDict(
                    [
                        (k.replace("module", "model"), v) if "module" in k else (k, v)
                        for k, v in checkpoint_dict["model_state_dict"].items()
                    ]
                )
            )
        else:
            print("Running on fully trained model...")

        self._log_param_sum(self.model, "after checkpoint, before ablation")

        # if reset or resample weights
        self.model = self.reset_resample_param(model=self.model)

        self._log_param_sum(self.model, "after ablation")

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        self._embed_dim, self._max_emb_layer = CARP_INFO[self._encoder_name]

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

    def _reset_weights(self, model: torch.nn.Module):
        print(f"Updating carp {self._encoder_name} weights...")
        for layer_name, p in model.state_dict().items():
            if "layers" in layer_name:
                if "conv" in layer_name:
                    if "weight" in layer_name:
                        kaiming_uniform_(p, a=math.sqrt(5))
                    elif "bias" in layer_name:
                        fan_in, _ = _calculate_fan_in_and_fan_out(
                            model.state_dict()[
                                layer_name.replace("bias", "weight")
                            ]
                        )
                        if fan_in != 0:
                            bound = 1 / math.sqrt(fan_in)
                            uniform_(p, -bound, bound)

                else:
                    if "weight" in layer_name:
                        ones_(p)
                    elif "bias" in layer_name:
                        zeros_(p)

    def _resample_weights(self, model: torch.nn.Module):
        print(f"Updating carp {self._encoder_name} weights...")
        resample_state = model.state_dict()
        for layer_name, p in model.state_dict().items():
            # completely shuffle all weight matrix entries
            if "layers" in layer_name:
                resample_state[layer_name] = p.view(-1)[
                    torch.randperm(p.view(-1).shape[0])
                ].view(p.shape)
        model.load_state_dict(resample_state)

    def _encode_batch(
        self,
        mut_seqs: Sequence[int] | int,
        dataset: Dataset,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encodes a batch of mutant sequences.

        Args:
        - mut_seqs: list of str or str, mutant sequences of the same length
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - np.ndarray or a tuple(np.ndarray, list[str]) where the list is batch_labels
        """
        if isinstance(mut_seqs, int):
            mut_seqs = [mut_seqs]

        mut_seqs = [[dataset.sequence[m]] for m in mut_seqs]

        x = self.collater(mut_seqs)[0].to(DEVICE)
        rep = self.model(x, repr_layers=list(range(self._max_emb_layer + 1)))

        # init output dict
        dict_encoded_mut_seqs = {}

        dict_encoded_mut_seqs[0] = self.flatten_encode(
            encoded_mut_seqs=rep[0].detach().cpu().numpy(),
            flatten_emb=flatten_emb,
            mut_seqs=mut_seqs,
        )

        for layer_numb, encoded_mut_seqs in rep["representations"].items():
            dict_encoded_mut_seqs[layer_numb] = self.flatten_encode(
                encoded_mut_seqs=encoded_mut_seqs.detach().cpu().numpy(),
                flatten_emb=flatten_emb,
                mut_seqs=mut_seqs,
            )
        return dict_encoded_mut_seqs


class ContextEncoder(AbstractEncoder):
    """
    Base class for query/context (MSA-conditioned) autoregressive encoders:
    ProtxLSTM, ProtMamba and PoET.

    These share the same batch protocol (a ``query`` sequence embedded in the
    context of a set of homologs), the same CLS-trimming + pooling tail, and the
    query-dict pooling length.  Model loading and the forward pass itself remain
    encoder-specific.
    """

    def _get_query_context(self, dataset, mut_seqs):
        """Split each dataset item into its query and context halves."""
        query = [dataset[m][0] for m in mut_seqs]
        context = [dataset[m][1] for m in mut_seqs]
        return query, context

    def _pooling_seq_len(self, mut_seq, encoded_mut_seq) -> int:
        # protmamba / poet query dict -> exclude CLS token
        if len(mut_seq) == 3:
            return len(mut_seq["input_ids"]) - 1
        return super()._pooling_seq_len(mut_seq, encoded_mut_seq)

    def _finalize_representation(self, representation, query, flatten_emb, mut_seqs):
        """Trim CLS, flatten/pool per layer, then run the reproducibility check."""
        # init output dict
        dict_encoded_mut_seqs = {}

        for layer_numb, encoded_mut_seqs in representation.items():

            # trim off initial classification token [CLS]
            if self._iftrimCLS:
                encoded_mut_seqs = encoded_mut_seqs[:, 1:, :]

            dict_encoded_mut_seqs[layer_numb] = self.flatten_encode(
                encoded_mut_seqs=encoded_mut_seqs,
                flatten_emb=flatten_emb,
                mut_seqs=query,
            )

        return dict_encoded_mut_seqs


class ProtxLSTMEncoder(ContextEncoder):
    """
    Build a ProtxLSTM encoder
    """

    def __init__(
        self,
        encoder_name: str,
        checkpoint: float = 1,
        checkpoint_folder: str = "pretrain_checkpoints/protxlstm",
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
        max_sequence_length: int = None,
        chunk_chunk_size: int = 2**14,
        mlstm_chunksize: int = 1024,
        iftrimCLS: bool = True,
        **kwargs,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of CARP_INFO
        - checkpoint: float = 1, the 0.5, 0.25, 0.125 checkpoint of the CARP encoder or full
        - checkpoint_folder: str = "pretrain_checkpoints/protxlstm", folder for carp encoders
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """
        _require(xLSTMLMHeadModel, "protxlstm", _PROTXLSTM_ERR)
        _require(ProteinDataCollator, "dataloaders", _DATALOADERS_ERR)

        super().__init__(encoder_name, reset_param, resample_param, embed_torch_seed)
        self.state = None
        self._iftrimCLS = iftrimCLS

        print(f"Seed for ProtxLSTMEncoder: {self._embed_torch_seed}")

        self.collater = ProteinDataCollator(max_sequence_length)

        config_update_kwargs = {
            "mlstm_backend": "chunkwise_variable",
            "mlstm_chunksize": mlstm_chunksize,
            "mlstm_return_last_state": True,
        }
        self.chunk_chunk_size = chunk_chunk_size

        self.model = load_model(
            f"models/{encoder_name}",
            model_class=xLSTMLMHeadModel,
            device="cpu",
            dtype=torch.float32,
            **config_update_kwargs,
        ).eval()

        self._log_param_sum(self.model, "after load, before ablation")

        # load checkpoint unless default to full
        if checkpoint != 1:
            # get the checkpoint number from the PROTXLSTM_CHECKPOINTS dict
            # ie {"protxlstm_102M_60B": {"1/2": 3_T16384, ...}, ...}
            # to get 'pretrain_checkpoints/protxlstm/3_T16384'

            checkpoint_path = (
                f"{os.path.normpath(checkpoint_folder)}/"
                f"{str(PROTXLSTM_CHECKPOINTS[encoder_name][checkpoint])}"
            )

            print(
                f"Loading {encoder_name} {checkpoint} checkpoint from {checkpoint_path}..."
            )

            self.model = load_model(
                checkpoint_path,
                model_class=xLSTMLMHeadModel,
                device="cpu",
                dtype=torch.float32,
                **config_update_kwargs,
            ).eval()

        else:
            print("Running on fully trained model...")

        self._log_param_sum(self.model, "after checkpoint, before ablation")

        # if reset or resample weights
        if self._reset_param:
            model_config = OmegaConf.load(
                "./src/models/protxlstm/configs/xlstm_default_config.yaml"
            )

            config_update_kwargs = {
                "backend": "chunkwise_variable",
                "chunk_size": mlstm_chunksize,
                "return_last_state": True,
            }

            if encoder_name == "protxlstm_26M_30B":
                model_config["model"]["embedding_dim"] = 512

            model_config["model"]["mlstm_block"]["mlstm"].update(config_update_kwargs)

            xlstm_config = xLSTMConfig().init_from_dict(model_config["model"])
            seed_all(self._embed_torch_seed)
            self.model = xLSTMLMHeadModel(xlstm_config)

        else:
            self.model = self.reset_resample_param(model=self.model)

        self._log_param_sum(self.model, "after ablation")

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        self._embed_dim, self._max_emb_layer = PROTXLSTM_INFO[self._encoder_name]

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

    def _reset_weights(self, model: torch.nn.Module):
        # NOTE: kept as-is for reproducibility. The `rand` ablation for ProtxLSTM
        # rebuilds a fresh model in __init__, so this hook is not reached on that
        # path; it is retained to mirror the original dispatch.
        print(f"Updating protxlstm {self._encoder_name} weights...")
        model.backbone.reset_parameters()

    def _resample_weights(self, model: torch.nn.Module):
        print(f"Updating protxlstm {self._encoder_name} weights...")
        resample_state = model.state_dict()
        for layer_name, p in model.state_dict().items():
            if "xlstm_block_stack" in layer_name:
                resample_state[layer_name] = p.view(-1)[
                    torch.randperm(p.view(-1).shape[0])
                ].view(p.shape)
        model.load_state_dict(resample_state)

    def _pooling_seq_len(self, mut_seq, encoded_mut_seq) -> int:
        # protxlstm query dict -> CLS token already excluded upstream
        if len(mut_seq) == 3:
            return encoded_mut_seq.shape[0]
        return super()._pooling_seq_len(mut_seq, encoded_mut_seq)

    def precompute_context_state(self, context, chunk_chunk_size=2**14):
        """
        Precompute the output states for a fixed context that remains the same across generations.
        Returns the hidden states to continue generation later.
        """
        state = None
        if isinstance(context, dict):
            context = [context]

        x = self.collater(context)

        input_ids = x["input_ids"]
        pos_ids = x["position_ids"]

        for chunk in range(input_ids.shape[1] // chunk_chunk_size + 1):

            start_idx = chunk * chunk_chunk_size
            end_idx = min((chunk + 1) * chunk_chunk_size, input_ids.shape[1])

            if start_idx == end_idx:
                pass

            else:
                input_ids_chunk = input_ids[:, start_idx:end_idx].to(DEVICE)
                pos_ids_chunk = pos_ids[:, start_idx:end_idx].to(DEVICE)

                with torch.no_grad():
                    outputs, _ = self.model(
                        input_ids=input_ids_chunk,
                        position_ids=pos_ids_chunk,
                        state=state,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    state = outputs.state

        # Return the hidden states for reuse
        return state

    def _encode_batch(
        self,
        mut_seqs: Sequence[int],
        dataset: DownstreamMemmapDataset,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encodes a batch of mutant sequences.

        Args:
        - mut_seqs: list of int or int, idx of sequences
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - np.ndarray or a tuple(np.ndarray, list[str]) where the list is batch_labels
        """

        query, context = self._get_query_context(dataset, mut_seqs)

        state = self.state

        if dataset.yield_with_context:
            state = self.precompute_context_state(
                context, chunk_chunk_size=self.chunk_chunk_size
            )

        x = self.collater(query)
        input_ids = x["input_ids"]
        pos_ids = x["position_ids"]

        # unflattened rep is {layer: (batch_size, seq_len+1, emb_dim)}
        representation = {
            layer: np.empty((len(mut_seqs), input_ids.shape[1], self._embed_dim))
            for layer in range(self._max_emb_layer + 1)
        }

        for chunk in range(input_ids.shape[1] // self.chunk_chunk_size + 1):

            start_idx = chunk * self.chunk_chunk_size
            end_idx = min((chunk + 1) * self.chunk_chunk_size, input_ids.shape[1])

            if start_idx == end_idx:
                pass

            else:
                input_ids_chunk = input_ids[:, start_idx:end_idx].to(DEVICE)
                pos_ids_chunk = pos_ids[:, start_idx:end_idx].to(DEVICE)

                with torch.no_grad():
                    outputs, rep = self.model(
                        input_ids=input_ids_chunk,
                        position_ids=pos_ids_chunk,
                        save_layer=list(range(self._max_emb_layer + 1)),
                        state=state,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    state = outputs.state
                    for layer in range(self._max_emb_layer + 1):
                        representation[layer][
                            :,
                            chunk
                            * self.chunk_chunk_size : (chunk + 1)
                            * self.chunk_chunk_size,
                            :,
                        ] = (
                            rep[layer].detach().cpu().numpy()
                        )

        return self._finalize_representation(
            representation, query, flatten_emb, mut_seqs
        )


class ProtMambaEncoder(ContextEncoder):
    """
    Build a ProtMamba encoder
    """

    def __init__(
        self,
        encoder_name: str,
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
        max_sequence_length: int = None,
        iftrimCLS: bool = True,
        chunk_chunk_size: int = 2**15,
        **kwargs,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of CARP_INFO
        - checkpoint: float = 1, the 0.5, 0.25, 0.125 checkpoint of the CARP encoder or full
        - checkpoint_folder: str = "pretrain_checkpoints/protxlstm", folder for carp encoders
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """
        _require(MambaLMHeadModelwithPosids, "protmamba", _PROTMAMBA_ERR)
        _require(load_model, "protxlstm", _PROTXLSTM_ERR)  # shared generic loader
        _require(ProteinDataCollator, "dataloaders", _DATALOADERS_ERR)

        super().__init__(encoder_name, reset_param, resample_param, embed_torch_seed)
        self.state = None
        self._iftrimCLS = iftrimCLS
        self.inference_param = InferenceParams

        print(f"Seed for ProtMambaEncoder: {self._embed_torch_seed}")

        self.collater = ProteinDataCollator(max_sequence_length)
        self.max_sequence_length = max_sequence_length
        self.chunk_chunk_size = chunk_chunk_size

        self.model = load_model(
            f"models/{encoder_name}",
            model_class=MambaLMHeadModelwithPosids,
            device="cuda",
            dtype=torch.float32,
            checkpoint_mixer=True,
        ).eval()

        self._log_param_sum(self.model, "after load, before ablation")

        print("Running on fully trained model...")

        # if reset or resample weights
        if self._reset_param:
            model_config = OmegaConf.load(
                "src/models/mamba/configs/default_config.yaml"
            )

            if encoder_name == "protmamba_107M_195B":
                model_config["model"]["d_model"] = 1024

            # Create new mode
            mamba_config = MambaConfig(
                d_model=model_config["model"]["d_model"],
                n_layer=model_config["model"]["n_layer"],
                vocab_size=model_config["model"]["vocab_size"],
                residual_in_fp32=model_config["model"]["residual_in_fp32"],
            )

            seed_all(self._embed_torch_seed)
            self.model = MambaLMHeadModelwithPosids(
                mamba_config, dtype=torch.float32, checkpoint_mixer=True
            )

        else:
            self.model = self.reset_resample_param(
                model=self.model
            )  # TODO: implement restructure

        self._log_param_sum(self.model, "after ablation")

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        self._embed_dim, self._max_emb_layer = PROTMAMBA_INFO[self._encoder_name]

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

    def _resample_weights(self, model: torch.nn.Module):
        # ProtMamba structure:
        # (backbone): MixerModelWithPosids( ... layers.n.mixer.ckpt_layer.Mamba(
        #     in_proj, conv1d, x_proj, dt_proj, out_proj ) ... )
        # => resample every linear/conv1d layer
        print(f"Updating protmamba {self._encoder_name} weights...")
        resample_state = model.state_dict()
        keywords = ["in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"]
        for layer_name, p in model.state_dict().items():
            if any(k in layer_name for k in keywords):
                resample_state[layer_name] = p.view(-1)[
                    torch.randperm(p.view(-1).shape[0])
                ].view(p.shape)
        model.load_state_dict(resample_state)

    def _encode_batch(
        self,
        mut_seqs: Sequence[int],
        dataset: DownstreamMemmapDataset,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encodes a batch of mutant sequences.

        Args:
        - mut_seqs: list of int or int, idx of sequences
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - np.ndarray or a tuple(np.ndarray, list[str]) where the list is batch_labels
        """
        query, context = self._get_query_context(dataset, mut_seqs)

        _query = self.collater(query)
        query_tokens = _query["input_ids"]
        query_pos_ids = _query["position_ids"]

        _context = self.collater(context)
        context_tokens = _context["input_ids"]
        context_pos_ids = _context["position_ids"]

        input_ids = torch.cat([context_tokens, query_tokens], dim=1).to(DEVICE)
        pos_ids = torch.cat([context_pos_ids, query_pos_ids], dim=1).to(DEVICE)

        inference_params = self.inference_param(
            max_seqlen=1, max_batch_size=input_ids.shape[0]
        )

        torch.cuda.empty_cache()
        with torch.no_grad():
            representation = self.model(
                input_ids[:, : self.chunk_chunk_size],
                position_ids=pos_ids[:, : self.chunk_chunk_size],
                inference_params=inference_params,
                save_layer=list(range(self._max_emb_layer + 1)),
            )

            if input_ids.shape[1] > self.chunk_chunk_size:
                rep = dict()
                for layer in range(self._max_emb_layer + 1):
                    rep[layer] = []
                # save any query tokens if included in the first chunk
                if context_tokens.shape[1] < self.chunk_chunk_size:
                    for layer in range(self._max_emb_layer + 1):
                        rep[layer].append(
                            representation[layer][
                                :, -(self.chunk_chunk_size - context_tokens.shape[1]) :
                            ]
                        )

                inference_params.seqlen_offset += self.chunk_chunk_size

                for i in tqdm(
                    range(self.chunk_chunk_size, input_ids.shape[1]),
                    desc="Process Sequence Rest",
                    total=len(range(self.chunk_chunk_size, input_ids.shape[1])),
                    leave=False,
                ):
                    representation = self.model(
                        input_ids[:, i : i + 1],
                        position_ids=pos_ids[:, i : i + 1],
                        inference_params=inference_params,
                        save_layer=list(range(self._max_emb_layer + 1)),
                    )
                    inference_params.seqlen_offset += 1
                    if i >= context_tokens.shape[1]:
                        for layer in range(self._max_emb_layer + 1):
                            rep[layer].append(representation[layer][:, -1:])

                representation = dict()
                for layer in range(self._max_emb_layer + 1):
                    representation[layer] = np.concatenate(rep[layer], axis=1)
            else:
                # layer representations have shape (batch_size, context_pos_ids.shape[1], emb_dim) => need to be cropped
                for layer in range(self._max_emb_layer + 1):
                    representation[layer] = representation[layer][
                        :, -query_tokens.shape[1] :
                    ]

        return self._finalize_representation(
            representation, query, flatten_emb, mut_seqs
        )


class PoETEncoder(ContextEncoder):
    """
    Build a PoETEncoder encoder
    """

    def __init__(
        self,
        encoder_name: str,
        reset_param: bool = False,
        resample_param: bool = False,
        embed_torch_seed: int = RAND_SEED,
        max_sequence_length: int = None,
        iftrimCLS: bool = True,
        iftrimEOS: bool = True,
    ):
        """
        Args
        - encoder_name: str, the name of the encoder, one of the keys of CARP_INFO
        - checkpoint: float = 1, the 0.5, 0.25, 0.125 checkpoint of the CARP encoder or full
        - checkpoint_folder: str = "pretrain_checkpoints/protxlstm", folder for carp encoders
        - reset_param: bool = False, if update the full model to xavier_uniform_
        - resample_param: bool = False, if update the full model to xavier_normal_
        """
        _require(PoET, "poet", _POET_ERR)
        _require(ProteinDataCollator, "dataloaders", _DATALOADERS_ERR)

        super().__init__(encoder_name, reset_param, resample_param, embed_torch_seed)
        self.state = None
        self._iftrimCLS = iftrimCLS
        self._iftrimEOS = iftrimEOS
        self.alphabet = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        )

        print(f"Seed for PoETEncoder: {self._embed_torch_seed}")

        self.collater = ProteinDataCollator(
            max_sequence_length, padding_value=self.alphabet.mask_token
        )

        self.max_sequence_length = max_sequence_length

        self.model = load_poet_model(
            f"models/{encoder_name}.ckpt",
            model_class=PoET,
            device="cuda",
            dtype=torch.bfloat16,
        ).eval()

        self._log_param_sum(self.model, "after load, before ablation")

        print("Running on fully trained model...")

        # if reset or resample weights
        if self._reset_param or self._resample_param:
            self.model = self.reset_resample_param(model=self.model)

        self._log_param_sum(self.model, "after ablation")

        # set model to eval mode
        self.model.eval()
        self.model.to(DEVICE)

        self._embed_dim, self._max_emb_layer = POET_INFO[self._encoder_name]

        # load model from torch.hub
        print(
            f"Generating {self._encoder_name} upto {self._max_emb_layer} layer embedding ..."
        )

    def _reset_weights(self, model: torch.nn.Module):
        """
        Re-initialise PoET weights.

        ``reset_param`` may be a string to restrict the reset to a subset of the
        attention:
        - ``"intra_seq"``: only attention applied to sequences independently
        - ``"inter_seq"``: only attention applied across sequences
        - otherwise: full re-initialisation following the scheme below.

        token_embed.weight                                  [nn.init.normal_(self.weight)]
        decoder.layers.n.self_attn.{q,k,v}_proj.weight      [xavier_uniform_(gain=1/sqrt(2))]
        decoder.layers.n.self_attn.out_proj.{weight,bias}   [constant_(0.0)]
        decoder.layers.n.multihead_attn.{q,k,v}_proj.weight [xavier_uniform_(gain=1/sqrt(2))]
        decoder.layers.n.multihead_attn.out_proj.{weight,bias} [constant_(0.0)]
        decoder.layers.n.linear1.weight / linear.weight     [kaiming_uniform_(a=sqrt(5))]
        decoder.layers.n.linear1.bias / linear.bias         [uniform_(-bound, bound)]
        decoder.layers.n.linear2.{weight,bias}              [constant_(0.0)]
        norm(.n).weight                                     [constant_(1.0)]
        norm(.n).bias                                       [constant_(0.0)]
        """
        print(f"Updating poet {self._encoder_name} weights...")
        if self._reset_param == "intra_seq":
            # only reset attention applied to sequences independently
            for layer_name, p in model.state_dict().items():
                if "self_attn" in layer_name and "_proj" in layer_name:
                    if "out" in layer_name or "bias" in layer_name:
                        # for both weight and bias
                        constant_(p, 0.0)
                    elif "weight" in layer_name:
                        xavier_uniform_(p, gain=1 / math.sqrt(2))

        elif self._reset_param == "inter_seq":
            # only reset attention applied across sequences
            for layer_name, p in model.state_dict().items():
                if "multihead_attn" in layer_name and "_proj" in layer_name:
                    if "out" in layer_name or "bias" in layer_name:
                        # for both weight and bias
                        constant_(p, 0.0)
                    elif "weight" in layer_name:
                        xavier_uniform_(p, gain=1 / math.sqrt(2))
        else:
            for layer_name, p in model.state_dict().items():
                if "linear2" in layer_name:
                    # for both weights and bias
                    constant_(p, 0.0)
                elif "token_embed" in layer_name:
                    normal_(p)
                elif "linear" in layer_name:
                    # use default init from nn.Linear:
                    if "weight" in layer_name:
                        kaiming_uniform_(p, a=math.sqrt(5))
                    elif "bias" in layer_name:
                        fan_in, _ = _calculate_fan_in_and_fan_out(
                            model.state_dict()[
                                layer_name.replace("bias", "weight")
                            ]
                        )
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        uniform_(p, -bound, bound)

                elif "norm" in layer_name:
                    # assume element-wise affine
                    if "weight" in layer_name:
                        constant_(p, 1.0)
                    elif "bias" in layer_name:
                        constant_(p, 0.0)

                # now, only attention layers should be
                elif "_proj" in layer_name:
                    if "out" in layer_name or "bias" in layer_name:
                        # for both weight and bias
                        constant_(p, 0.0)
                    elif "weight" in layer_name:
                        xavier_uniform_(p, gain=1 / math.sqrt(2))

    def _resample_weights(self, model: torch.nn.Module):
        print(f"Updating poet {self._encoder_name} weights...")
        resample_state = model.state_dict()
        # don't shuffle inv_freq bc not trainable!
        for layer_name, p in model.state_dict().items():
            if "decoder" in layer_name and not "rotary_emb" in layer_name:
                resample_state[layer_name] = p.view(-1)[
                    torch.randperm(p.view(-1).shape[0])
                ].view(p.shape)
        model.load_state_dict(resample_state)

    def update_max_sequence_length(self, max_sequence_length):
        self.collater.max_sequence_length = max_sequence_length
        self.max_sequence_length = max_sequence_length

    def precompute_context_state(self, context, chunk_chunk_size=None):
        encoded_context, segment_sizes = self.alphabet.translate(context["input_ids"])
        if len(segment_sizes) < 1:
            return None

        segment_sizes = torch.tensor(segment_sizes).to(DEVICE)
        encoded_context = encoded_context.long().to(DEVICE)

        memory = self.model.embed(
            encoded_context.unsqueeze(0),
            segment_sizes.unsqueeze(0),
            pbar_position=None,
        )
        return memory

    def _encode_batch(
        self,
        mut_seqs: Sequence[int],
        dataset: DownstreamMemmapDataset,
        flatten_emb: bool | str,
        mut_names: Sequence[str] | str | None = None,
    ) -> np.ndarray:
        """
        Encodes a batch of mutant sequences.

        Args:
        - mut_seqs: list of int or int, idx of sequences
        - flatten_emb: bool or str, if and how (one of ["max", "mean"]) to flatten the embedding
        - mut_names: list of str or str or None, mutant names

        Returns:
        - np.ndarray or a tuple(np.ndarray, list[str]) where the list is batch_labels
        """

        assert len(mut_seqs) == 1, "PoETEncoder does not yet support batches > 1"

        torch.cuda.empty_cache()

        query, context = self._get_query_context(dataset, mut_seqs)

        memory = self.state

        if dataset.yield_with_context:
            memory = self.precompute_context_state(context[0])

        query_tokens, _ = self.alphabet.translate(query[0]["input_ids"])
        query_translated = [{"input_ids": query_tokens.long()}]

        _query_translated = self.collater(query_translated)

        input_ids = _query_translated["input_ids"].to(DEVICE)

        is_memory_preallocated = self.state is not None

        with torch.no_grad():
            _, representation = self.model.logits(
                input_ids,
                memory,
                preallocated_memory=is_memory_preallocated,
                save_layer=list(range(self._max_emb_layer + 1)),
            )

        for layer in range(self._max_emb_layer + 1):
            representation[layer] = representation[layer].detach().cpu().float().numpy()

        return self._finalize_representation(
            representation, query, flatten_emb, mut_seqs
        )


def get_emb_info(encoder_name: str) -> Collection(str, AbstractEncoder, int):
    """
    A function return processed encoder_name and total_emb_layer

    Args:
    - encoder_name: str, input encoder_name

    Returns:
    - encoder_name: str, change anything not a transformer or carp encoder to onehot
    - encoder_class: AbstractEncoder, encoder class
    - total_emb_layer: int, number of embedding layers
    """

    if encoder_name in TRANSFORMER_INFO.keys():
        total_emb_layer = TRANSFORMER_INFO[encoder_name][1] + 1
        encoder_class = ESMEncoder
    elif encoder_name in CARP_INFO.keys():
        total_emb_layer = CARP_INFO[encoder_name][1] + 1
        encoder_class = CARPEncoder
    elif encoder_name in PROTXLSTM_INFO.keys():
        total_emb_layer = PROTXLSTM_INFO[encoder_name][1] + 1
        encoder_class = ProtxLSTMEncoder
    elif encoder_name in PROTMAMBA_INFO.keys():
        total_emb_layer = PROTMAMBA_INFO[encoder_name][1] + 1
        encoder_class = ProtMambaEncoder
    elif encoder_name in POET_INFO.keys():
        total_emb_layer = POET_INFO[encoder_name][1] + 1
        encoder_class = PoETEncoder
    else:
        # for onehot
        encoder_name = "onehot"
        encoder_class = OnehotEncoder
        total_emb_layer = 1

    return encoder_name, encoder_class, total_emb_layer
