import numpy as np
import cv2
from typing import Optional
from skimage.measure import label, regionprops
from . import utils
from .base import ImageSegmentator


class ThresholdImageSegmentator(ImageSegmentator):
    def __init__(self, global_threshold=5, regional_threshold=None, global_kernel_size=23, regional_kernel_size=13, pixel_to_length_ratio = np.sqrt(0.08), volume_threshold=10):
        super().__init__()
        self.global_threshold = global_threshold
        self.regional_threshold = regional_threshold
        self.global_kernel_size = global_kernel_size
        self.regional_kernel_size = regional_kernel_size
        self.pixel_to_length_ratio = pixel_to_length_ratio
        self.volume_threshold = volume_threshold
        self.object_counter = 1  # Counter for assigning unique values to objects

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
        masks = utils.image_to_masks(masks)

        return masks

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

    def _calculate_region_volume(self, region):
        minr, minc, maxr, maxc = region.bbox

        # Calculate the width and height of the bounding box
        width = self.pixel_to_length_ratio * (maxc - minc)
        height = self.pixel_to_length_ratio * (maxr - minr)

        volume = np.pi * width ** 2 * height / 6

        return volume

    def _regional_to_global_mask(self, region_masks):
        # Combine the region masks using logical OR operatioimg_normn
        global_mask = np.zeros_like(region_masks[0], dtype=np.uint8)
        for mask in region_masks:
            global_mask = cv2.bitwise_or(global_mask, mask)

        return global_mask

    def _per_region_segmentation(self, image, mask):
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)       

        region_masks = []
        for region in regions:
            #volume = region.area
            volume = self._calculate_region_volume(region)
            if volume < self.volume_threshold:
                continue

            initial_region_mask = self._get_region_mask(labeled_mask, region)
            masked_region = image * initial_region_mask

            if self.regional_threshold is None:
                region_mask = self._apply_threshold(masked_region, -1, self.regional_kernel_size, use_otsu=True)
            else:
                region_mask = self._apply_threshold(masked_region, self.regional_threshold, self.regional_kernel_size)

            region_mask_filled = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            region_mask_filled = np.where(region_mask_filled == 255, self.object_counter, region_mask_filled)
            self.object_counter += 1

            region_masks.append(region_mask_filled)

        global_mask = self._regional_to_global_mask(region_masks)
        self.object_counter = 1
        return global_mask

