import numpy as np
from typing import List

def flatten_masks(masks: List[np.ndarray]) -> np.ndarray:
    return [m.flatten().tolist() for m in masks]

def image_to_masks(image):
    """
    Converts a labeled image to a list of binary masks.
    """
    unique_labels = np.unique(image)  # Get the unique labels in the image

    masks = []
    for label in unique_labels:
        if label == 0:
            continue  # Skip the background label

        mask = np.where(image == label, 255, 0).astype(np.uint8)  # Convert to CV_8UC1 format
        masks.append(mask)

    return masks
