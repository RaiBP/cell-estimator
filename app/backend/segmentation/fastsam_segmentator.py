import numpy as np
import logging
from typing import Union

import sys

from .FastSAM.fastsam import FastSAM, FastSAMPrompt

from . import config
from .base import ImageSegmentator

logging.basicConfig(level=logging.INFO)


class FastSAMImageSegmentator(ImageSegmentator):
    def __init__(self):
        super().__init__()
        self.device = config.FASTSAM["DEVICE"]
        self.model = FastSAM(config.FASTSAM["FASTSAM_MODEL_PATH"])
        self.results = None

    def set_image(self, image: np.ndarray) -> np.ndarray:
        logging.info("[FastSAM] Generating embeddings")
        self.image = image
        return self.model(
            self.image, device=self.device, retina_masks=True, conf=0.4, iou=0.9
        )[0]    # get only the first result since we only have one image here

    def segment(self, image: np.ndarray) -> np.ndarray:
        logging.info("[FastSAM] Segmenting image")
        self.results = self.set_image(image)
        masks = np.array([mask.data.cpu().numpy() for mask in self.results.masks]).squeeze()
        masks = (masks * 255.).astype(np.uint8)
        return masks

    def prompt(self, query: dict = None) -> np.ndarray:
        raise NotImplementedError("FastSAM does not support querying yet")

    def name(self) -> str:
        return "fastsam"
