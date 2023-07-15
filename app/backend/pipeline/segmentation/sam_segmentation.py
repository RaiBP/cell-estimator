from segmentation.segmentation import Segmentation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np

class SAMSegmentation(Segmentation):
    def __init__(self, use_phase=True):
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.model = SamAutomaticMaskGenerator(sam)
        self.use_phase = use_phase


    def _segment_single_image(self, phase, amplitude):
        img = phase if self.use_phase else amplitude

        img = self._preprocess_sam(img)

        masks_img = self.model.generate(img)
        return masks_img
    
    def _list_of_outlines(self, masks):
        pass
    
    def _grayscale_to_rgb(self, img):
        return np.stack([img, img, img], axis=-1)
    
    def _normalize_minmax(self, img):
        return img - np.min(img) / (np.max(img) - np.min(img))
    
    def _preprocess_sam(self, img):
        img = 255 * self._normalize_minmax(img)
        img = img.astype(np.uint8)
        img = self._grayscale_to_rgb(img)
        return img
    
