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

def calculate_perimeter(mask, length_to_pixel_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0
    for contour in contours:
        perimeter += cv2.arcLength(contour, True) 
    return perimeter * length_to_pixel_ratio


def calculate_area(num_pixels, area_to_pixel_ratio):
    return area_to_pixel_ratio * num_pixels


def calculate_volume(mask, length_to_pixel_ratio):
    indices = np.where(mask != 0)

    # Get the minimum and maximum coordinates
    min_x, max_x = np.min(indices[1]), np.max(indices[1])
    min_y, max_y = np.min(indices[0]), np.max(indices[0])

    # Calculate the width and height of the mask
    width = (max_x - min_x + 1) * length_to_pixel_ratio
    height = (max_y - min_y + 1) * length_to_pixel_ratio

    return width ** 2 * height * np.pi / 6


def calculate_roundness(area, perimeter):
    return 4 * np.pi * area / perimeter ** 2
   

def calculate_opacity(masked_normalized_amplitude, area):
    return np.sum(1 - masked_normalized_amplitude) / area


def calculate_mean(masked_image, pixels_in_mask):
    return np.sum(masked_image) / pixels_in_mask


def calculate_variance(masked_image, mean, pixels_in_mask):
    return np.sum((masked_image - mean) ** 2) / (pixels_in_mask - 1)


def calculate_skewness(masked_image, mean, pixels_in_mask):
    return (np.sum((masked_image - mean) ** 3) / pixels_in_mask) / (np.sum((masked_image - mean) ** 2) / pixels_in_mask) ** 1.5


def calculate_kurtosis(masked_image, mean, pixels_in_mask):
    return (np.sum((masked_image - mean) ** 4) / pixels_in_mask) / (np.sum((masked_image - mean) ** 2) / pixels_in_mask) ** 2


def calculate_masked_image(image, binary_mask):
    # returns a list of values
    return image[binary_mask]

def calculate_centroid_coordinates(masked_image, num_pixels):
    grid_x, grid_y = np.meshgrid(np.arange(masked_image.shape[1]), np.arange(masked_image.shape[0]))
    centroid_x = np.sum(grid_x * masked_image) / num_pixels
    centroid_y = np.sum(grid_y * masked_image) / num_pixels
    return centroid_x, centroid_y

def calculate_centroid_displacement(centroid_x_1, centroid_y_1, centroid_x_2, centroid_y_2, length_to_pixel_ratio):
    diff_x = centroid_x_1 - centroid_x_2
    diff_y = centroid_y_1 - centroid_y_2
    return np.sqrt(diff_x ** 2 + diff_y ** 2) * length_to_pixel_ratio

def calculate_dry_mass(phase_mean, wavelength, refractive_increment, area_to_pixel_ratio, num_pixels):
    # in picograms
    return area_to_pixel_ratio * wavelength  * phase_mean * num_pixels / (1e3 * 2 * np.pi * refractive_increment)


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

def calculate_masked_image_keeping_dimensions(image, mask): 
    # returns an actual image
    binary_mask = np.where(mask != 0, 1, 0).astype(np.uint8)  # Convert mask to 0-1 binary mask
    return image * binary_mask


def calculate_std_kernel(masked_image, kernel_size):
    mean_kernel = apply_kernel(masked_image, kernel_size)

    kernel_squared = mean_kernel ** 2 
    squared_kernel = apply_kernel(masked_image ** 2, kernel_size)
    kernel_pixels = kernel_size ** 2 
    diff = squared_kernel - kernel_squared
    diff[diff<0] = 0
    std_kernel = np.sqrt((kernel_pixels / (kernel_pixels - 1)) * diff)
    std_kernel = replace_nan_with_min_value(std_kernel)

    #std_kernel = np.pad(std_kernel, ((0, 1), (0, 1)), mode='constant')
    return std_kernel 


def img_to_uint8(img):
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)


def replace_nan_with_min_value(image):
    min = np.nanmin(image)
    image[np.isnan(image)] = min
    return image


def kernel_size_micrometer_to_pixels(kernel_size_in_micrometers, length_to_pixel_ratio):
    kernel_size_in_pixels = int(np.round(kernel_size_in_micrometers / length_to_pixel_ratio))

    # we want an odd integer for kernel size
    return kernel_size_in_pixels if kernel_size_in_pixels % 2 == 0 else kernel_size_in_pixels + 1
