import numpy as np
from typing import List

def flatten_masks(masks: List[np.ndarray]) -> np.ndarray:
    return [m.flatten().tolist() for m in masks]
