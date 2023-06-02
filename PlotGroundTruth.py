import h5py
import matplotlib.pyplot as plt

# Open the HDF5 file
file = h5py.File(r"W:\prediction.seg", 'r')

# Access the dataset
dataset = file['phase/images']

labels = file['label/ground_truth']

# Choose the images you want to display
image_indices = [1, 10, 30]  # Replace with the indices of the images you want to display

# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Display the selected images
for i, index in enumerate(image_indices):
    image = dataset[index, :, :]
    axes[i].imshow(image)
    axes[i].set_title(f'{labels[index]}')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

# Close the HDF5 file
file.close()