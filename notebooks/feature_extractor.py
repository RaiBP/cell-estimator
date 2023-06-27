import numpy as np
import pandas as pd
from feature_extraction import * 

class FeatureExtractor:
    def __init__(self, phase_array, amplitude_array, mask_array, pixel_to_area_ratio=0.08, wavelength=530, refractive_increment=0.2, kernel_size=5):
        self.batch = list(zip(phase_array, amplitude_array, mask_array))
        self.pixel_to_area_ratio = pixel_to_area_ratio
        self.pixel_to_length_ratio = np.sqrt(pixel_to_area_ratio)
        self.wavelength = wavelength
        self.refractive_increment = refractive_increment
        self.kernel_size = kernel_size
        self.features = None

        self.phase_mean_kernel_array = []
        self.phase_std_kernel_array = []


    def extract_features(self):
        # Loop over each element of the batch and extract features
        columns = ['Volume', 'Roundness', 'Opacity', 'Amplitude Variance', 'Amplitude Skewness', 'Dry Mass Density', 'Max Phase', 'Phase Variance', 'Phase Skewness', 'DC1', 'DC2', 'DC3']
        extracted_features = pd.DataFrame(columns=columns)

        for index, (phase, amplitude, masks) in enumerate(self.batch):
            for mask in masks:
                features_single_cell = self.extract_features_single_image(index, phase, amplitude, mask)
                features_df = pd.DataFrame([features_single_cell], columns=columns)
                extracted_features = pd.concat([extracted_features, features_df], ignore_index=True)
        # Return the extracted features for all elements of the batch
        self.features = extracted_features
        return extracted_features


    def extract_features_single_image(self, index, phase, amplitude, mask):
        binary_mask = calculate_binary_mask(mask)
        num_pixels = calculate_num_pixels(mask)
        volume = calculate_volume(mask, self.pixel_to_length_ratio)
        perimeter = calculate_perimeter(mask, self.pixel_to_length_ratio)
        area = calculate_area(num_pixels, self.pixel_to_area_ratio)
        roundness = calculate_roundness(area, perimeter)
        masked_normalized_amplitude = calculate_masked_normalized_amplitude(amplitude, binary_mask)
        opacity = calculate_opacity(masked_normalized_amplitude, area)
        amplitude_mean = calculate_amplitude_mean(masked_normalized_amplitude, num_pixels)
        amplitude_variance = calculate_amplitude_variance(masked_normalized_amplitude, amplitude_mean, num_pixels)
        amplitude_skewness = calculate_amplitude_skewness(masked_normalized_amplitude, amplitude_mean, num_pixels)
        masked_phase = calculate_masked_phase(phase, binary_mask)
        max_phase = calculate_max_phase(masked_phase)
        phase_mean = calculate_phase_mean(masked_phase, num_pixels)
        phase_variance = calculate_phase_variance(masked_phase, phase_mean, num_pixels)
        phase_skewness = calculate_phase_skewness(masked_phase, phase_mean, num_pixels)
        dry_mass = calculate_dry_mass(phase_mean, self.wavelength, self.refractive_increment, self.pixel_to_area_ratio, num_pixels)
        dry_mass_density = calculate_dry_mass_density(dry_mass, volume)

        masked_phase_original_values = calculate_masked_phase_with_original_values(phase, mask)
        phase_mean_kernel = calculate_phase_mean_within_kernel(masked_phase_original_values, self.kernel_size)
        phase_std_kernel = calculate_phase_std_within_kernel(masked_phase_original_values, phase_mean_kernel, self.kernel_size)     
        self.phase_mean_kernel_array.append(phase_mean_kernel)
        self.phase_std_kernel_array.append(phase_std_kernel)
        dc1 = calculate_dry_mass_density_contrast_1(phase_std_kernel, num_pixels)
        dc2 = calculate_dry_mass_density_contrast_2(phase_std_kernel, dc1, num_pixels)
        dc3 = calculate_dry_mass_density_contrast_3(phase_std_kernel, dc1, num_pixels)
        return volume, roundness, opacity, amplitude_variance, amplitude_skewness, dry_mass_density, max_phase, phase_variance, phase_skewness, dc1, dc2, dc3



    
