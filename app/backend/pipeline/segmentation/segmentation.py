import numpy as np 
from abc import ABC, abstractmethod

from cellpose_segmentation import CellposeSegmentation
from sam_segmentation import SAMSegmentation
from threshold_segmentation import ThresholdSegmentation

class Segmentation(ABC):
    def __init__(self):

    def segment(self, phase, amplitude):
        mask_image = self._segment_single_image(phase, amplitude)
        return self._image_to_masks(mask_image)
    

    def outlines(self, mask):
        return self._list_of_outlines(mask)


    @staticmethod
    def create_model(selector):
        if selector == "cellpose":
            return CellposeSegmentation()
        elif selector == "threshold":
            return ThresholdSegmentation()
        elif selector == "sam":
            return SAMSegmentation()
        else:
            raise ValueError("Invalid segmentation model")


    @staticmethod
    def _image_to_masks(image):
        unique_labels = np.unique(image)  # Get the unique labels in the image

        masks = []
        for label in unique_labels:
            if label == 0:
                continue  # Skip the background label

            mask = np.where(image == label, 255, 0).astype(np.uint8)  # Convert to CV_8UC1 format
            masks.append(mask)

        return masks


    @abstractmethod
    def _segment_single_image(self, phase, amplitude):
        pass


    @abstractmethod
    def _list_of_outlines(self, mask):
        pass
