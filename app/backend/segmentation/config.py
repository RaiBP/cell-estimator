import torch

IMAGE_TYPE = "phase"

TORCH_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MOBILE_SAM = {
    "CHECKPOINT_PATH": "/home/fidelinus/Downloads/mobile_sam.pt",
    "MODEL_TYPE": "vit_t",
    "DEVICE": TORCH_DEVICE,
}

FASTSAM = {
    "FASTSAM_MODEL_PATH": "/home/fidelinus/Downloads/FastSAM-x.pt",
    "DEVICE": TORCH_DEVICE,
}

SAM = {
    "CHECKPOINT_PATH": "/home/fidelinus/Downloads/sam_vit_b_01ec64.pth",
    "MODEL_TYPE": "vit_b",
    "DEVICE": TORCH_DEVICE,
}

CELLPOSE = {
    "MODEL_KWARGS": {
        "model_type": "cyto",
        "gpu": True,
    },
    "PREDICTION_KWARGS": {
        "diameter": None,
        "channels": [0, 0],
        "flow_threshold": 0.4,
        "do_3D": False,
        "cellprob_threshold": -0.0,
        "min_size": -1,
        "augment": True,
        "net_avg": True,
        "resample": True,
    },
}
