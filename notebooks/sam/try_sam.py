import torch
import torchvision
import cv2
import sys
import numpy as np
import h5py
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from pathlib import Path

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def grayscale_to_rgb(image):
    return np.stack([image, image, image], axis=-1)

def normalize_minmax(img):
    return img - np.min(img) / (np.max(img) - np.min(img))

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

# Open the HDF5 file
DATA_FOLDER = Path("/home/fidelinus/tum/applied_machine_intelligence/final_project/data")
filepath = DATA_FOLDER / "sample01.pre"
file = h5py.File(filepath, 'r')

# Access the dataset
# List dataset in the file
amplitude = file['amplitude/images']
phase = file['phase/images']

img_ix = 0
phase_img = phase[img_ix, :, :]
amplitude_img = amplitude[img_ix, :, :]
print(phase_img.shape)

# masks = file['mask/images']

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

phase_img = grayscale_to_rgb(phase_img)
phase_img = 255 * normalize_minmax(phase_img)
phase_img = phase_img.astype(np.uint8)
mask_generator = SamAutomaticMaskGenerator(sam)
# predictor = SamPredictor(sam)
# predictor.set_image(phase_img)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

masks = mask_generator.generate(phase_img)

plt.imshow(phase_img)
show_anns(masks)
plt.show()
