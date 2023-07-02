import torch
import cellpose

IMAGE_TYPE = "phase"

TORCH_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

FASTSAM = {
    "FASTSAM_MODEL_PATH": "/home/fidelinus/Downloads/FastSAM-x.pt",
    "DEVICE": TORCH_DEVICE,
}

CELLPOSE = {
    "MODEL_KWARGS": {
        "model_type": "cyto",
        "gpu": False,
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
    }
}
