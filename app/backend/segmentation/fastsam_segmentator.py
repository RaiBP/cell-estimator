import numpy as np
import logging
from typing import Union

import sys

sys.path.append("../FastSAM")
from fastsam import FastSAM, FastSAMPrompt

import utils
import config
from base import ImageSegmentator

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

        self.results_cache[image_id] = self.model(self.image, device=self.device, retina_masks=True, conf=0.4, iou=0.9)
        self.results = self.results_cache[image_id]

    def segment(self, query=None) -> np.ndarray:
        if self.results is None:
            raise RuntimeError("Image not set. Please call set_image() first.")

        self.prompt = FastSAMPrompt(self.image, self.results, device=self.device)

        if query is None:
            masks = [r.masks.xyn for r in self.results]
            masks = utils.flatten_masks(masks[0])
            return masks
        else:
            raise NotImplementedError("Querying is not implemented yet.")
