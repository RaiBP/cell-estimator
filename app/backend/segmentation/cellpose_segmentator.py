import numpy as np
from cellpose import models
from typing import Union

import config
from base import ImageSegmentator


class CellPoseImageSegmentator(ImageSegmentator):
    def __init__(self):
        super().__init__()
        self.model = models.Cellpose(**config.CELLPOSE["MODEL_KWARGS"])

    def set_image(self, image: np.ndarray, image_id: Union[str, int]) -> np.ndarray:
        self.image = image

    def segment(self, query=None) -> np.ndarray:
        masks, _, _, _ = self.model.eval(self.image, **config.CELLPOSE["PREDICTION_KWARGS"])
        print(masks)
