import numpy as np
import logging
from abc import abstractmethod
from typing import Optional


logging.basicConfig(level=logging.INFO)


def compute_array_hash(array: np.ndarray) -> str:
    return hash(array.tobytes())


def cache_embeddings(func):
    def wrapper(self, image: np.ndarray):
        image_hash = compute_array_hash(image)
        if image_hash in self.embeddings_cache:
            logging.info("Embeddings cache hit")
            return self.embeddings_cache[image_hash]
        else:
            logging.info("Embeddings cache miss")
            embeddings = func(self, image)
            self.embeddings_cache[image_hash] = embeddings
            return embeddings

    return wrapper


def cache_masks(func):
    def wrapper(self, image: np.ndarray):
        image_hash = compute_array_hash(image)
        if image_hash in self.masks_cache:
            logging.info("Masks cache hit")
            return self.masks_cache[image_hash]
        else:
            masks = func(self, image)
            logging.info("Masks cache miss")
            self.masks_cache[image_hash] = masks
            return masks

    return wrapper


class ImageSegmentator:
    def __init__(self):
        self.image = None
        self.embeddings_cache = {}
        self.masks_cache = {}

    @cache_embeddings
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        return self.set_image(image)

    @cache_masks
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        return self.segment(image)

    @abstractmethod
    def set_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Receives an image and generates predictions/embeddings for it
        Returns the embeddings result, in whatever format the algorithm uses
        """
        pass

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Receives an image and its id and returns an array of binary masks, which has shame (N, H, W)
        """
        pass

    def prompt(self, query: dict = None) -> np.ndarray:
        """
        Used to query the segmentation algorithm
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the algorithm
        """
        pass
