from segmentation import Segmentation
import numpy as np
import cv2
from skimage.measure import label, regionprops

class ThresholdSegmentation(Segmentation):
    def __init__(self, phase_array, amplitude_array, global_threshold=5, regional_threshold=None, global_kernel_size=23, regional_kernel_size=13, pixel_to_length_ratio = np.sqrt(0.08), volume_threshold=10, use_phase_global_thresholding=False, use_phase_regional_thresholding=False):
        super().__init__(phase_array, amplitude_array)
        self.global_threshold = global_threshold
        self.regional_threshold = regional_threshold
        self.global_kernel_size = global_kernel_size
        self.regional_kernel_size = regional_kernel_size
        self.pixel_to_length_ratio = pixel_to_length_ratio
        self.volume_threshold = volume_threshold
        self.use_phase_global_thresholding = use_phase_global_thresholding
        self.use_phase_regional_thresholding = use_phase_regional_thresholding
        self.object_counter = 1  # Counter for assigning unique values to objects
    
    def _segment_single_image(self, phase, amplitude):
        image_for_global_thresholding = phase if self.use_phase_global_thresholding else amplitude
        image_for_regional_thresholding = phase if self.use_phase_regional_thresholding else amplitude

        if self.global_threshold is None:
            initial_mask = self._apply_threshold(image_for_global_thresholding, -1, self.global_kernel_size, use_otsu=True)
        else:
            initial_mask = self._apply_threshold(image_for_global_thresholding, self.global_threshold, self.global_kernel_size)

        final_mask = self._per_region_segmentation(image_for_regional_thresholding, initial_mask)

        return final_mask
    

    def _list_of_outlines(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines = list(contours)

        for i in range(len(outlines)):
            outlines[i] = outlines[i].reshape(-1, 2)

        return outlines
        

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
    def _crop_square_region(image, minr, maxr, minc, maxc, square_size):
        # Crop the square region
        square_image = image[minr:maxr, minc:maxc]

        # Pad the square region with zeros if necessary
        pad_height = max(0, square_size - square_image.shape[0])
        pad_width = max(0, square_size - square_image.shape[1])
        square_image = np.pad(square_image, ((0, pad_height), (0, pad_width)), mode='constant')
        return square_image

    def _extract_square_region(self, image, region_mask, region, margin=10, min_square_size=100):
        # Calculate the bounding box coordinates
        minr, minc, maxr, maxc = region.bbox

        # Calculate the width and height of the bounding box
        width = maxc - minc
        height = maxr - minr

        # Calculate the size of the square region
        size = max(width, height) + margin * 2

        # Adjust the size to match the desired square size
        size = max(size, min_square_size)

        # Calculate the center coordinates of the bounding box
        center_row = (minr + maxr) // 2
        center_col = (minc + maxc) // 2

        # Calculate the new bounding box coordinates for the square region
        minr = center_row - size // 2
        minc = center_col - size // 2
        maxr = minr + size
        maxc = minc + size

        # Ensure the coordinates are within image bounds
        minr = max(0, minr)
        minc = max(0, minc)
        maxr = min(image.shape[0], maxr)
        maxc = min(image.shape[1], maxc)

        # Extract the region of interest
        masked_image = region_mask * image

        square_masked_image = self._crop_square_region(masked_image, minr, maxr, minc, maxc, size)

        return square_masked_image

    def _get_region_mask(self, labeled_mask, region):
        # Create a binary mask for the region
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
            volume = self._calculate_region_volume(region)
            # Ignore regions that are too small to be a cell
            if volume < self.volume_threshold:
                continue

            initial_region_mask = self._get_region_mask(labeled_mask, region)
            masked_region = image * initial_region_mask

            if self.regional_threshold is None:
                region_mask = self._apply_threshold(masked_region, -1, self.regional_kernel_size, use_otsu=True)
            else:
                region_mask = self._apply_threshold(masked_region, self.regional_threshold, self.regional_kernel_size)

            # Fill holes inside the region mask
            region_mask_filled = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            region_mask_filled = np.where(region_mask_filled == 255, self.object_counter, region_mask_filled)
            self.object_counter += 1

            region_masks.append(region_mask_filled)

        global_mask = self._regional_to_global_mask(region_masks)
        self.object_counter = 1
        return global_mask


    @staticmethod
    def _image_to_uint8(image):
        img_pos = np.abs(image)
        img_norm = ((img_pos - np.min(img_pos)) / (np.max(img_pos) - np.min(img_pos)) * 255).astype(np.uint8)
        return img_norm
