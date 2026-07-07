# Original code from Protein-Transfer under MIT License.

# Copyright (c) Microsoft Corporation. 

# Modified by Antonia Ebner, 2026:
#   - extended for protxlstm, protmamba and poet
#   - added Model_Info and Arch_Info as dataclasses

"""Embedding constants"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

ARCH_TYPE = ["carp", "esm"]
ARCH_TYPE_CLM = ["protxlstm", "protmamba", "poet"]  
ARCH_TYPE_LABELS = ["Prot-xLSTM", "ProtMamba", "PoET"]

ARCH_AB_DICT = {"rand": "random init", "stat": "stat transfer"}
ARCH_AB = sorted(deepcopy(list(ARCH_AB_DICT.keys())))

ARCH_BAR_LAYER = [0, 2, 4, 6]

ARCH_CUT_DICT = {"": [2, 3, 4, 6], "carp": [2, 4, 6, 12], "esm": [2, 3, 4, 6]}

MAX_SEQ_LEN = 2048  

@dataclass
class Model_Info:
    """Dataclass for infos about different models"""

    name: str
    arch: str
    n_param: int  # in M
    n_layers: int
    embed_dim: int
    size_category: str
    checkpoints: tuple[float] = None
    checkpoint_names: dict = None
    checkpoint_losses: dict = None


@dataclass
class Arch_Info:
    """Dataclass for infos about different architectures"""

    name: str
    models: Dict[str, Model_Info]
    large_model: str
    pretrain_task: str


PROTMAMBA_MODEL_INFO = Arch_Info(
    name="protmamba",
    large_model="protmamba_107M_195B",
    pretrain_task="clm",
    models={
        "protmamba_28M_30B": Model_Info(
            name="protmamba_28M_30B",
            arch="protmamba",
            n_param=28,
            n_layers=16,
            embed_dim=512,
            size_category="Small",
        ),
        "protmamba_107M_195B": Model_Info(
            name="protmamba_107M_195B",
            arch="protmamba",
            n_param=107,
            n_layers=16,
            embed_dim=1024,
            size_category="Large",
        ),
    },
)

POET_MODEL_INFO = Arch_Info(
    name="poet",
    large_model="poet_201M",
    pretrain_task="clm",
    models={
        "poet_201M": Model_Info(
            name="poet_201M",
            arch="poet",
            n_param=201,
            n_layers=12,
            embed_dim=1024,
            size_category="Large",
        ),
    },
)

PROTXLSTM_MODEL_INFO = Arch_Info(
    name="protxlstm",
    large_model="protxlstm_102M_60B",
    pretrain_task="clm",
    models={
        "protxlstm_26M_30B": Model_Info(
            name="protxlstm_26M_30B",
            arch="protxlstm",
            n_param=26,
            n_layers=16,
            embed_dim=512,
            size_category="Small",
        ),
        "protxlstm_102M_60B": Model_Info(
            name="protxlstm_102M_60B",
            arch="protxlstm",
            n_param=102,
            n_layers=16,
            embed_dim=1024,
            size_category="Large",
            checkpoints=(1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125),
            checkpoint_names={
                0.125: "0_T2048",
                0.25: "1_T4096",
                0.375: "2_T8192",
                0.5: "3_T16384",
                0.625: "4_T32768",
                0.75: "5_T65536",
                0.875: "6_T131072",
            },
            checkpoint_losses={
                0.125: 2.3195407390594482,
                0.25: 2.2641046047210693,
                0.375: 2.2123701572418213,
                0.5: 2.158830404281616,
                0.625: 2.119004726409912,
                0.75: 2.033418893814087,
                0.875: 1.9276466369628906,
                1: 1.8841066360473633,
            },
        ),
    },
)

CARP_MODEL_INFO = Arch_Info(
    name="carp",
    large_model="carp_640M",
    pretrain_task="mlm",
    models={
        "carp_600k": Model_Info(
            name="carp_600k",
            arch="carp",
            n_param=0.6,
            n_layers=16,
            embed_dim=128,
            size_category="Tiny",
            checkpoints=(1, 0.5, 0.25, 0.125),
            checkpoint_names={
                0.125: 52039,
                0.25: 114344,
                0.5: 239263,
            },
            checkpoint_losses={
                0.125: 2.5268538140067247,
                0.25: 2.517630364015801,
                0.5: 2.5123053303486858,
                1: 2.5051483969586483,
            },
        ),
        "carp_38M": Model_Info(
            name="carp_38M",
            arch="carp",
            n_param=38,
            n_layers=16,
            embed_dim=1024,
            size_category="Small",
            checkpoints=(1, 0.5, 0.25, 0.125),
            checkpoint_names={
                0.125: 129575,
                0.25: 256897,
                0.5: 517622,
            },
            checkpoint_losses={
                0.125: 2.3630260306320774,
                0.25: 2.338586726549564,
                0.5: 2.3189422531432275,
                1: 2.3030167711945997,
            },
        ),
        "carp_76M": Model_Info(
            name="carp_76M",
            arch="carp",
            n_param=76,
            n_layers=32,
            embed_dim=1024,
            size_category="Medium",
            checkpoints=(1, 0.5, 0.25, 0.125),
            checkpoint_names={
                0.125: 83180,
                0.25: 162959,
                0.5: 327960,
            },
            checkpoint_losses={
                0.125: 2.277597215778113,
                0.25: 2.248077081483563,
                0.5: 2.224771098829285,
                1: 2.2056100474366542,
            },
        ),
        "carp_640M": Model_Info(
            name="carp_640M",
            arch="carp",
            n_param=640,
            n_layers=56,
            embed_dim=1280,
            size_category="Large",
            checkpoints=(1, 0.5, 0.25, 0.125),
            checkpoint_names={
                0.125: 78810,
                0.25: 154698,
                0.5: 311757,
            },
            checkpoint_losses={
                0.125: 2.1455246604919718,
                0.25: 2.094229900229448,
                0.5: 2.0535276480066376,
                1: 2.0194828284458466,
            },
        ),
    },
)

ESM_MODEL_INFO = Arch_Info(
    name="esm",
    large_model="esm1b_t33_650M_UR50S",
    pretrain_task="mlm",
    models={
        "esm1_t6_43M_UR50S": Model_Info(
            name="esm1_t6_43M_UR50S",
            arch="esm",
            n_param=43,
            n_layers=6,
            embed_dim=768,
            size_category="Small",
        ),
        "esm1_t12_85M_UR50S": Model_Info(
            name="esm1_t12_85M_UR50S",
            arch="esm",
            n_param=85,
            n_layers=12,
            embed_dim=768,
            size_category="Medium",
        ),
        "esm1_t34_670M_UR50S": Model_Info(
            name="esm1_t34_670M_UR50S",
            arch="esm",
            n_param=670,
            n_layers=34,
            embed_dim=1280,
            size_category="Large",
        ),
        "esm1b_t33_650M_UR50S": Model_Info(
            name="esm1b_t33_650M_UR50S",
            arch="esm",
            n_param=650,
            n_layers=33,
            embed_dim=1280,
            size_category="Large+",
        ),
    },
)

PROTMAMBA_INFO = {
    "protmamba_28M_30B": (512, 16),
    "protmamba_107M_195B": (1024, 16),
}

# encoder_name: (d_model, n_layers)
PROTXLSTM_INFO = {
    "protxlstm_26M_30B": (512, 16),
    "protxlstm_102M_60B": (1024, 16),
}

TRANSFORMER_INFO = {
    "esm1_t6_43M_UR50S": (768, 6, 2),
    "esm1_t12_85M_UR50S": (768, 12, 2),
    "esm1_t34_670M_UR50S": (1280, 34, 2),
    "esm1b_t33_650M_UR50S": (1280, 33, 2),
}

# encoder_name: (d_model, n_layers)
CARP_INFO = {
    "carp_600k": (128, 16),
    "carp_38M": (1024, 16),
    "carp_76M": (1024, 32),
    "carp_640M": (1280, 56),
}

POET_INFO = {
    "poet_201M": (1024, 12),
}

# model parameter number in M
MODEL_SIZE = {
    "esm1_t6_43M_UR50S": 43,
    "esm1_t12_85M_UR50S": 85,
    "esm1_t34_670M_UR50S": 670,
    "esm1b_t33_650M_UR50S": 650,
    "carp_600k": 0.6,
    "carp_38M": 38,
    "carp_76M": 76,
    "carp_640M": 640,
    "protxlstm_26M_30B": 26,
    "protxlstm_102M_60B": 102,
    "protmamba_28M_30B": 28,
    "protmamba_107M_195B": 107,
    "poet_201M": 201,
    "onehot": 0.02,
}

EMB_SIMPLE_MAP = {
    "esm1_t6_43M_UR50S": "ESM1 43M",
    "esm1_t12_85M_UR50S": "ESM1 85M",
    "esm1_t34_670M_UR50S": "ESM1 670M",
    "esm1b_t33_650M_UR50S": "ESM1b 650M",
    "carp_600k": "CARP 0.6M",
    "carp_38M": "CARP 38M",
    "carp_76M": "CARP 76M",
    "carp_640M": "CARP 640M",
    "onehot": "Onehot",
}

EMB_SIZE_SIMPLE = {
    "esm1_t6_43M_UR50S": "Small",
    "esm1_t12_85M_UR50S": "Medium",
    "esm1_t34_670M_UR50S": "Large",
    "esm1b_t33_650M_UR50S": "Large*",
    "carp_600k": "Mini",
    "carp_38M": "Small",
    "carp_76M": "Medium",
    "carp_640M": "Large",
}

BASELINE_NAME_DICT = {
    "onehot": "One-hot",
    "rand": "Random init",
    "stat": "Stat transfer",
}
EMB_SIZE_NAME_SIMPLE = ["Small", "Medium", "Large"]
EMB_SIZE_NAME_SIMPLE_CLM = ["Small", "Large"]

# the embeddings to use for task oriented plots
EMB4TASK = [
    "onehot",
    "esm1_t6_43M_UR50S",
    "esm1_t12_85M_UR50S",
    "esm1b_t33_650M_UR50S",
    "carp_38M",
    "carp_76M",
    "carp_640M",
    "protxlstm_26M_30B",
    "protxlstm_102M_60B",
    "protmamba_28M_30B",
    "protmamba_107M_195B",
    "poet_201M",
]

# emb model parameter number in M
EMB_MODEL_SIZE = {k: v for k, v in deepcopy(MODEL_SIZE).items() if k != "onehot"}

MODEL_LAYER = {
    model_name: model_dets[1]
    for info_dict in [
        deepcopy(TRANSFORMER_INFO),
        deepcopy(CARP_INFO),
        deepcopy(PROTXLSTM_INFO),
        deepcopy(PROTMAMBA_INFO),
        deepcopy(POET_INFO),
        {"onehot": (1, 1)},
    ]
    for model_name, model_dets in info_dict.items()
}

EMB_MODEL_LAYER = {k: v for k, v in deepcopy(MODEL_LAYER).items() if k != "onehot"}

CARP_MODEL_LAYER = {k: v[-1] for k, v in deepcopy(CARP_INFO).items()}

CHECKPOINT_PERCENT = [0.125, 0.25, 0.5, 1]

# TODO integrate to be auto from sheet
PROTXLSTM_CHECKPOINTS = {
    "protxlstm_102M_60B": {
        0.125: "0_T2048",
        0.25: "1_T4096",
        0.375: "2_T8192",
        0.5: "3_T16384",
        0.625: "4_T32768",
        0.75: "5_T65536",
        0.875: "6_T131072",
    },
}

PROTXLSTM_CHECKPOINT_LOSSES = {
    "protxlstm_102M_60B": {
        0.125: 2.3195407390594482,
        0.25: 2.2641046047210693,
        0.375: 2.2123701572418213,
        0.5: 2.158830404281616,
        0.625: 2.119004726409912,
        0.75: 2.033418893814087,
        0.875: 1.9276466369628906,
        1: 1.8841066360473633,
    },
}

CARP_CHECKPOINTS = {
    "carp_600k": {0.5: 239263, 0.25: 114344, 0.125: 52039},
    "carp_38M": {0.5: 517622, 0.25: 256897, 0.125: 129575},
    "carp_76M": {0.5: 327960, 0.25: 162959, 0.125: 83180},
    "carp_640M": {0.5: 311757, 0.25: 154698, 0.125: 78810},
}

CARP_CHECKPOINT_LOSSES = {
    "carp_600k": {
        1: 2.5051483969586483,
        0.5: 2.5123053303486858,
        0.25: 2.517630364015801,
        0.125: 2.5268538140067247,
    },
    "carp_38M": {
        1: 2.3030167711945997,
        0.5: 2.3189422531432275,
        0.25: 2.338586726549564,
        0.125: 2.3630260306320774,
    },
    "carp_76M": {
        1: 2.2056100474366542,
        0.5: 2.224771098829285,
        0.25: 2.248077081483563,
        0.125: 2.277597215778113,
    },
    "carp_640M": {
        1: 2.0194828284458466,
        0.5: 2.0535276480066376,
        0.25: 2.094229900229448,
        0.125: 2.1455246604919718,
    },
}
