import torch

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
