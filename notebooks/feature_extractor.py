import numpy as np
from tqdm import tqdm
import pandas as pd
from feature_extraction import * 

def kernel_size_micrometer_to_pixels(kernel_size_in_micrometers, length_to_pixel_ratio):
    kernel_size_in_pixels = int(np.round(kernel_size_in_micrometers / length_to_pixel_ratio))

    # we want an odd integer for kernel size
    return kernel_size_in_pixels if kernel_size_in_pixels % 2 == 0 else kernel_size_in_pixels + 1

class FeatureExtractor:
    def __init__(self, phase_array, amplitude_array, mask_array, area_to_pixel_ratio=0.08, wavelength=530, refractive_increment=0.2, kernel_size=1): 
        """
        Kernel size of 1 μm for the STD filter and 2μm for the entropy filter taken from the supplementary material of [1]. 
        Formulas for entropy features also taken from the supplementary material of [1].

        References:
        [1] Siu, D. M. D., Lee, K. C. M., Lo, M. C. K., Stassen, S. V., Wang, M., Zhang, I. Z. Q., So, H. K. H., Chan, G. C. F., Cheah, K. S. E., Wong, K. K. Y., Hsin, M. K. Y., Ho, J. C. M., & Tsia, K. K. (2020). Deep-learning-assisted biophysical imaging cytometry at massive throughput delineates cell population heterogeneity. Lab on a Chip, 20(20), 3696–3708. https://doi.org/10.1039/D0LC00542H
        """
        self.batch = list(zip(phase_array, amplitude_array, mask_array))
        self.area_to_pixel_ratio = area_to_pixel_ratio
        self.length_to_pixel_ratio = np.sqrt(area_to_pixel_ratio)
        self.wavelength = wavelength
        self.refractive_increment = refractive_increment

        # from micrometers to pixels
        self.kernel_size = kernel_size_micrometer_to_pixels(kernel_size, self.length_to_pixel_ratio)

        self.features = None

        self.columns = ['Volume', 'Roundness', 'Opacity', 'AmplitudeVariance', 'AmplitudeSkewness', 'MaxAmplitude', 'MinAmplitude', 'DryMassDensity', 'MaxPhase', 'MinPhase', 'PhaseVariance', 'PhaseSkewness', 'PhaseSTDLocalMean', 'PhaseSTDLocalVariance', 'PhaseSTDLocalSkewness', 'PhaseSTDLocalKurtosis', 'PhaseSTDLocalMin', 'PhaseSTDLocalMax', 'PhaseSTDLocalCentroidDisplacement','AmplitudeSTDLocalMean', 'AmplitudeSTDLocalVariance', 'AmplitudeSTDLocalSkewness', 'AmplitudeSTDLocalKurtosis', 'AmplitudeSTDLocalMin', 'AmplitudeSTDLocalMax', 'AmplitudeSTDLocalCentroidDisplacement']
        

    def extract_features(self):
        # Loop over each element of the batch and extract features
        extracted_features = pd.DataFrame(columns=self.columns)

        for index, (phase, amplitude, mask) in enumerate(self.batch):
            features_single_cell = self.extract_features_single_image(index, phase, amplitude, mask)
            features_df = pd.DataFrame([features_single_cell], columns=self.columns)
            extracted_features = pd.concat([extracted_features, features_df], ignore_index=True)
        # Return the extracted features for all elements of the batch
        self.features = extracted_features
        return extracted_features

    def extract_features_multiple_masks(self, image_ids=None):
        # Loop over each element of the batch and extract features
        if image_ids is not None:
            self.columns.append("Image ID")

        total_iterations = len(self.batch)
        extracted_features = pd.DataFrame(columns=self.columns)

        progress_bar = tqdm(total=total_iterations, desc="Extracting features", unit="image")

        for index, (phase, amplitude, masks) in enumerate(self.batch):
            for mask in masks:
                features_single_cell = self.extract_features_single_image(index, phase, amplitude, mask)
                if image_ids is not None:
                    features_single_cell.append(image_ids[index])
                features_df = pd.DataFrame([features_single_cell], columns=self.columns)
                extracted_features = pd.concat([extracted_features, features_df], ignore_index=True)
            progress_bar.update(1)

        progress_bar.close()
        # Return the extracted features for all elements of the batch
        self.features = extracted_features
        return extracted_features


    def extract_features_single_image(self, index, phase, amplitude, mask):
        binary_mask = calculate_binary_mask(mask)
        num_pixels = calculate_num_pixels(mask)
        volume = calculate_volume(mask, self.length_to_pixel_ratio)
        perimeter = calculate_perimeter(mask, self.length_to_pixel_ratio)
        area = calculate_area(num_pixels, self.area_to_pixel_ratio)
        roundness = calculate_roundness(area, perimeter)

        normalized_amplitude = amplitude / 255
        masked_normalized_amplitude = calculate_masked_image(normalized_amplitude, binary_mask)
        opacity = calculate_opacity(masked_normalized_amplitude, area)
        amplitude_mean = calculate_mean(masked_normalized_amplitude, num_pixels)
        amplitude_variance = calculate_variance(masked_normalized_amplitude, amplitude_mean, num_pixels)
        amplitude_skewness = calculate_skewness(masked_normalized_amplitude, amplitude_mean, num_pixels)
        max_amplitude = np.max(masked_normalized_amplitude)
        min_amplitude = np.min(masked_normalized_amplitude)

        masked_phase = calculate_masked_image(phase, binary_mask)
        max_phase = np.max(masked_phase)
        min_phase = np.min(masked_phase)
        phase_mean = calculate_mean(masked_phase, num_pixels)
        phase_variance = calculate_variance(masked_phase, phase_mean, num_pixels)
        phase_skewness = calculate_skewness(masked_phase, phase_mean, num_pixels)
        dry_mass = calculate_dry_mass(phase_mean, self.wavelength, self.refractive_increment, self.area_to_pixel_ratio, num_pixels)
        dry_mass_density = calculate_dry_mass_density(dry_mass, volume)

        mask_centroid_x, mask_centroid_y = calculate_centroid_coordinates(mask, num_pixels)

        masked_phase_original_dimensions = calculate_masked_image_keeping_dimensions(phase, mask)

        phase_std_kernel = calculate_std_kernel(masked_phase_original_dimensions, self.kernel_size)

        phase_std_local_mean, phase_std_local_std, phase_std_local_skewness, phase_std_local_kurtosis, phase_std_local_min, phase_std_local_max, phase_std_local_centroid_displacement = self._calculate_local_features(phase_std_kernel, mask, binary_mask, num_pixels, mask_centroid_x, mask_centroid_y)

        masked_amplitude_original_dimensions = calculate_masked_image_keeping_dimensions(normalized_amplitude, mask)

        amplitude_std_kernel = calculate_std_kernel(masked_amplitude_original_dimensions, self.kernel_size)

        amplitude_std_local_mean, amplitude_std_local_std, amplitude_std_local_skewness, amplitude_std_local_kurtosis, amplitude_std_local_min, amplitude_std_local_max, amplitude_std_local_centroid_displacement = self._calculate_local_features(amplitude_std_kernel, mask, binary_mask, num_pixels, mask_centroid_x, mask_centroid_y)

        feature_list = [volume, roundness, opacity, amplitude_variance, amplitude_skewness, max_amplitude, min_amplitude, dry_mass_density, max_phase, min_phase, phase_variance, phase_skewness, phase_std_local_mean, phase_std_local_std, phase_std_local_skewness, phase_std_local_kurtosis, phase_std_local_min, phase_std_local_max, phase_std_local_centroid_displacement,  amplitude_std_local_mean, amplitude_std_local_std, amplitude_std_local_skewness, amplitude_std_local_kurtosis, amplitude_std_local_min, amplitude_std_local_max, amplitude_std_local_centroid_displacement]

        return feature_list


    def _calculate_local_features(self, local_image, mask, binary_mask, num_pixels, mask_centroid_x, mask_centroid_y):
        masked_kernel = calculate_masked_image(local_image, binary_mask)
        masked_kernel_original_dimensions = calculate_masked_image_keeping_dimensions(local_image, mask)

        local_mean = calculate_mean(masked_kernel, num_pixels)
        local_std = calculate_variance(masked_kernel, local_mean, num_pixels)
        local_skewness = calculate_skewness(masked_kernel, local_mean, num_pixels)
        local_kurtosis = calculate_kurtosis(masked_kernel, local_mean, num_pixels)
        local_min = np.min(masked_kernel)
        local_max = np.max(masked_kernel)

        local_centroid_x, local_centroid_y = calculate_centroid_coordinates(masked_kernel_original_dimensions, num_pixels)
        local_centroid_displacement = calculate_centroid_displacement(local_centroid_x, local_centroid_y, mask_centroid_x, mask_centroid_y, self.length_to_pixel_ratio)

        return local_mean, local_std, local_skewness, local_kurtosis, local_min, local_max, local_centroid_displacement    
