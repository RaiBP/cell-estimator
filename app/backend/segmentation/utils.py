import cv2
import numpy as np
from cellpose import utils
from typing import List
from segmentation.threshold_segmentator import ThresholdImageSegmentator
from segmentation.fastsam_segmentator import FastSAMImageSegmentator
from segmentation.sam_segmentator import SAMImageSegmentator
from segmentation.cellpose_segmentator import CellPoseImageSegmentator

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512


def create_segmentation_model(selector):
    if selector == "cellpose":
        return CellPoseImageSegmentator()   
    elif selector == "threshold":
        return ThresholdImageSegmentator()
    elif selector == "sam":
        return SAMImageSegmentator()
    elif selector =="fastsam":
        return FastSAMImageSegmentator()
    else:
        raise ValueError("Invalid segmentation model")




def normalize_contour(contour: np.ndarray) -> np.ndarray:
    """
    Normalizes a contour to the range [0, 1].
    """
    contour = contour.astype(np.float32)
    contour[:, 0] /= IMAGE_WIDTH
    contour[:, 1] /= IMAGE_HEIGHT
    return contour


def get_mask_contour(mask: np.ndarray) -> np.ndarray:
    """
    Returns the contour of a binary mask.
    """
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    contours = contours[0].squeeze()
    contours = np.vstack((contours, contours[0]))  # close the contour
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

        mask = np.where(image == label, 255, 0).astype(
            np.uint8
        )  # Convert to CV_8UC1 format
        masks.append(mask)

    return masks
