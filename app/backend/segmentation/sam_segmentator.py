import numpy as np
import logging
from typing import Union, Optional
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from . import config
from .base import ImageSegmentator

logging.basicConfig(level=logging.INFO)

class SAMImageSegmentator(ImageSegmentator):
    def __init__(self):
        super().__init__()
        self._checkpoint = config.SAM["CHECKPOINT_PATH"]
        self._model_type = config.SAM["MODEL_TYPE"]
        self._device = config.SAM["DEVICE"]
        self._sam = sam_model_registry[self._model_type](checkpoint=self._checkpoint)
        self._sam.to(device=self._device)
        self.predictor = SamPredictor(self._sam)
        self.mask_generator = SamAutomaticMaskGenerator(self._sam)
        self.mask_generation_mode = True

    def set_image(self, image: np.ndarray, image_id: Union[str, int]) -> np.ndarray:
        self.image = image
        logging.info("Setting image for SAM - Generating embeddings")
        self.predictor.set_image(image)

    def segment(self, image: np.ndarray, image_id: Union[str, int]) -> np.ndarray:
        logging.info("Generating masks with SAM - Mask generator")
        results = self.mask_generator.generate(image)
        return np.array([r['segmentation'] for r in results])

    def prompt(self, query: Optional[dict] = None) -> np.ndarray:
        return self.predictor.predict(**query)
