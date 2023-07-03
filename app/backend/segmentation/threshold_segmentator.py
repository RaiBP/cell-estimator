import numpy as np
import cv2
from typing import Optional
from skimage.measure import label, regionprops

from .base import ImageSegmentator


class ThresholdImageSegmentator(ImageSegmentator):
    def __init__(self, global_threshold=5, regional_threshold=None, global_kernel_size=23, regional_kernel_size=13, volume_threshold=10):
        super().__init__()
        self.global_threshold = global_threshold
        self.regional_threshold = regional_threshold
        self.global_kernel_size = global_kernel_size
        self.regional_kernel_size = regional_kernel_size
        self.volume_threshold = volume_threshold

    def set_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        "Does nothing for this segmentator"
        pass

    def segment(self, image: np.ndarray) -> np.ndarray:

        # Convert to grayscale if needed
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.global_threshold is None:
            initial_mask = self._apply_threshold(image, -1, self.global_kernel_size, use_otsu=True)
        else:
            initial_mask = self._apply_threshold(image, self.global_threshold, self.global_kernel_size)

        masks = self._per_region_segmentation(image, initial_mask)

        return np.array(masks)

    def _apply_threshold(self, image, threshold, kernel_size, use_otsu=False):

        image_uint8 = self._image_to_uint8(image)
        blurred_image = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), 0)
        
        if use_otsu:
            _, mask_inv = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, mask_inv = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY_INV)

        mask = cv2.bitwise_not(mask_inv)

        return mask

    @staticmethod
    def _image_to_uint8(image):
        img_pos = np.abs(image)
        img_norm = ((img_pos - np.min(img_pos)) / (np.max(img_pos) - np.min(img_pos)) * 255).astype(np.uint8)
        return img_norm

    def _get_region_mask(self, labeled_mask, region):
        region_mask = np.zeros_like(labeled_mask)
        region_mask[labeled_mask == region.label] = 1
        return region_mask

    def _per_region_segmentation(self, image, mask):
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        region_masks = []
        for region in regions:
            volume = region.area
            if volume < self.volume_threshold:
                continue

            initial_region_mask = self._get_region_mask(labeled_mask, region)
            masked_region = image * initial_region_mask

            if self.regional_threshold is None:
                region_mask = self._apply_threshold(masked_region, -1, self.regional_kernel_size, use_otsu=True)
            else:
                region_mask = self._apply_threshold(masked_region, self.regional_threshold, self.regional_kernel_size)

            region_mask_filled = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            region_masks.append(region_mask_filled)

        return region_masks

