import numpy as np
import logging
from cellpose import models
from typing import Union

from . import utils
from . import config
from .base import ImageSegmentator

logging.basicConfig(level=logging.INFO)

class CellPoseImageSegmentator(ImageSegmentator):
    def __init__(self):
        super().__init__()
        self.model = models.Cellpose(**config.CELLPOSE["MODEL_KWARGS"])

    def set_image(self, image: np.ndarray) -> np.ndarray:
        logging.warning("CellPose does not have a special functionality for setting images. Use segment directly. - Skipping")
        pass

    def segment(self, image: np.ndarray) -> np.ndarray:
        masks, _, _, _ = self.model.eval(image, **config.CELLPOSE["PREDICTION_KWARGS"])
        masks = utils.image_to_masks(masks)
        return np.array(masks)

    def prompt(self, query: dict = None) -> np.ndarray:
        raise NotImplementedError("CellPose does not support prompting")
