# Original code from PoET under MIT License.
# Copyright (c) 2023 OpenProteinAI

import torch


def load_model(model_path, device, model_class, dtype=torch.bfloat16):
    ckpt = torch.load(model_path)
    model = model_class(**ckpt["hyper_parameters"]["model_spec"]["init_args"])
    model.load_state_dict(
        {k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()}
    )

    model = model.to(dtype).to(device).eval()
    return model
