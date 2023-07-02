import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class ImageSegmentator(ABC):

    def __init__(self):
        self.image = None

    @abstractmethod
    def set_image(self, image: np.ndarray, image_id: Union[str, int]) -> np.ndarray:
        """
        Receives an image and generates predictions/embeddings for it
        """
        pass

    @abstractmethod
    def segment(self, image: np.ndarray, image_id: Union[str, int]) -> np.ndarray:
        """
        Receives an image and its id and returns an array of binary masks, which has shame (N, H, W)
        """
        pass

    def prompt(self, query: dict = None) -> np.ndarray:
        """
        Used to query the segmentation algorithm
        """
        pass

