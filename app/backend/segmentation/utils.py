import cv2
import numpy as np
from cellpose import utils
from typing import List

def get_mask_contour(mask: np.ndarray) -> np.ndarray:
    """
    Returns the contour of a binary mask.
    """
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    contours = contours[0].squeeze()
    contours = np.vstack((contours, contours[0]))   # close the contour
    return contours

def flatten_contours(masks: List[np.ndarray]) -> np.ndarray:
    """
    Transforms a 2D array of contour coordinates into a 1D array, given following format
    of the input array:
    (n_points, 2) -> (n_points * 2) where the output array is arranged in (x1, y1, x2, y2, ...)
    """
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
