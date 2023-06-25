import numpy as np 
import cv2


def calculate_num_pixels(mask):
    """
    Returns the amount of pixels inside the mask
    We assume that the unselected pixels are zero
    """
    return cv2.countNonZero(mask)

def calculate_binary_mask(mask):
    return np.where(mask != 0, True, False)

def calculate_perimeter(mask, pixel_to_length_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.arcLength(contours[0], True) * pixel_to_length_ratio


def calculate_area(num_pixels, pixel_to_area_ratio):
    return pixel_to_area_ratio * num_pixels


def calculate_volume(mask, pixel_to_length_ratio):
    indices = np.where(mask != 0)

    # Get the minimum and maximum coordinates
    min_x, max_x = np.min(indices[1]), np.max(indices[1])
    min_y, max_y = np.min(indices[0]), np.max(indices[0])

    # Calculate the width and height of the mask
    width = (max_x - min_x + 1) * pixel_to_length_ratio
    height = (max_y - min_y + 1) * pixel_to_length_ratio

    return width ** 2 * height * np.pi / 6


def calculate_roundness(area, perimeter):
    return 4 * np.pi * area / perimeter ** 2
   

def calculate_masked_normalized_amplitude(amplitude, binary_mask):
    normalized_amplitude = amplitude / 255
    return normalized_amplitude[binary_mask]


def calculate_opacity(masked_normalized_amplitude, area):
    return np.sum(1 - masked_normalized_amplitude) / area


def calculate_mean(masked_image, pixels_in_mask):
    return np.sum(masked_image) / pixels_in_mask


def calculate_variance(masked_image, mean, pixels_in_mask):
    return np.sum((masked_image - mean) ** 2) / (pixels_in_mask - 1)


def calculate_skewness(masked_image, mean, pixels_in_mask):
    return (np.sum((masked_image - mean) ** 3) / pixels_in_mask) / (np.sum((masked_image - mean) ** 2) / pixels_in_mask) ** 1.5


def calculate_amplitude_mean(masked_normalized_amplitude, num_pixels):
    return calculate_mean(masked_normalized_amplitude, num_pixels)


def calculate_amplitude_variance(masked_normalized_amplitude, amplitude_mean, num_pixels):
    return calculate_variance(masked_normalized_amplitude, amplitude_mean, num_pixels)


def calculate_amplitude_skewness(masked_normalized_amplitude, amplitude_mean, num_pixels):
    return calculate_skewness(masked_normalized_amplitude, amplitude_mean, num_pixels)


def calculate_masked_phase(phase, binary_mask):
    return phase[binary_mask]


def calculate_max_phase(masked_phase):
    return np.max(np.abs(masked_phase))


def calculate_phase_mean(masked_phase, num_pixels):
    return calculate_mean(masked_phase, num_pixels)


def calculate_phase_variance(masked_phase, phase_mean, num_pixels):
    return calculate_variance(masked_phase, phase_mean, num_pixels)
    

def calculate_phase_skewness(masked_phase, phase_mean, num_pixels):
    return calculate_skewness(masked_phase, phase_mean, num_pixels)


def calculate_dry_mass(phase_mean, wavelength, refractive_increment, pixel_to_area_ratio, num_pixels):
    # in picograms
    return pixel_to_area_ratio * wavelength  * phase_mean * num_pixels / (1e3 * 2 * np.pi * refractive_increment)


def calculate_dry_mass_density(dry_mass, volume):
    return dry_mass / volume


def apply_kernel(image, kernel_size):
    image = image.astype(np.float32)
    # Define the kernel as a rectangular window of size r x r
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Pad the input image to handle border pixels
    image_padded = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REFLECT)

    # Apply the kernel to the padded image using filter2D
    result_padded = cv2.filter2D(image_padded, -1, kernel)

    # Crop the output image to remove the padding
    result = result_padded[kernel_size//2:-kernel_size//2, kernel_size//2:-kernel_size//2]
    
    return result

def calculate_masked_phase_with_original_values(phase, mask): 
    binary_mask = np.where(mask != 0, 1, 0).astype(np.uint8)  # Convert mask to 0-1 binary mask
    return phase * binary_mask


def calculate_phase_mean_within_kernel(masked_phase_original_values, kernel_size):
    return apply_kernel(masked_phase_original_values, kernel_size)


def calculate_phase_std_within_kernel(masked_phase_original_values, phase_mean_kernel, kernel_size):
    phase_kernel_squared = phase_mean_kernel ** 2 
    phase_squared_kernel = apply_kernel(masked_phase_original_values ** 2, kernel_size)
    kernel_pixels = kernel_size ** 2 
    dif = phase_squared_kernel - phase_kernel_squared
    dif[dif<0] = 0
    return np.sqrt((kernel_pixels / (kernel_pixels - 1)) * dif)


def calculate_dry_mass_density_contrast_1(phase_std_kernel, num_pixels):
    return calculate_mean(phase_std_kernel, num_pixels)


def calculate_dry_mass_density_contrast_2(phase_std_kernel, dc1, num_pixels):
    return calculate_variance(phase_std_kernel, dc1, num_pixels)


def calculate_dry_mass_density_contrast_3(phase_std_kernel, dc1, num_pixels):
    return calculate_skewness(phase_std_kernel, dc1, num_pixels)

