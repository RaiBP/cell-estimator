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
    def segment(self, query=None) -> np.ndarray:
        """
        Receives an image and returns an array of masks in normalized XY format
        """
        pass

