import numpy as np
import logging
from typing import Union

import sys

sys.path.append("../FastSAM")
from fastsam import FastSAM, FastSAMPrompt

from . import config
from .base import ImageSegmentator

logging.basicConfig(level=logging.INFO)


class FastSAMImageSegmentator(ImageSegmentator):
    def __init__(self):
        super().__init__()
        self.device = config.FASTSAM["DEVICE"]
        self.model = FastSAM(config.FASTSAM["FASTSAM_MODEL_PATH"])
        self.results = None
        self.results_cache = {}

    def set_image(self, image: np.ndarray, image_id: Union[str, int]) -> np.ndarray:
        self.image = image

        if image_id in self.results_cache:
            self.results = self.results_cache[image_id]
            logging.info(f"Image {image_id} found in cache.")
            return

        self.results_cache[image_id] = self.model(
            self.image, device=self.device, retina_masks=True, conf=0.4, iou=0.9
        )[0]    # get only the first result since we only have one image here
        self.results = self.results_cache[image_id]

    def segment(self, image: np.ndarray, image_id: Union[str, int]) -> np.ndarray:
        self.set_image(image, image_id)
        masks = np.array([mask.data.cpu().numpy() for mask in self.results.masks]).squeeze()
        return masks

    def prompt(self, query: dict = None) -> np.ndarray:
        raise NotImplementedError("FastSAM does not support querying yet")
