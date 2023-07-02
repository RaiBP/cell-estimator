from segmentation import Segmentation
import numpy as np

class SAMSegmentation(Segmentation):
    def __init__(self, phase_array, amplitude_array, model, use_phase=True):
        super().__init__(phase_array, amplitude_array)
        self.model = model
        self.use_phase = use_phase

    def _segment_single_image(self, phase, amplitude):
        # Implement SAM-based segmentation here
        # Return the segmented image


        img = phase if self.use_phase else amplitude

        img = self._preprocess_sam(img)

        masks_img = self.model.generate(img)

        return masks_img
    
    
    def _grayscale_to_rgb(self, img):
        return np.stack([img, img, img], axis=-1)

    def _normalize_minmax(self, img):
        return img - np.min(img) / (np.max(img) - np.min(img))
    
    def _preprocess_sam(self, img):
        img = 255 * self._normalize_minmax(img)
        img = img.astype(np.uint8)
        img = self._grayscale_to_rgb(img)
        return img

    def _list_of_outlines(self, masks):
        pass
