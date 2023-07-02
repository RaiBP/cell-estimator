import sys
sys.path.append("..")

from segmentation.fastsam_segmentator import FastSAMImageSegmentator
from image_loader import ImageLoader, prepare_amplitude_img, prepare_phase_img

loader = ImageLoader.from_file("/home/fidelinus/tum/applied_machine_intelligence/final_project/data/real_world_sample01.pre")

amplitude_img, phase_img = loader.get_images(0)
amplitude_img = prepare_amplitude_img(amplitude_img)
phase_img = prepare_phase_img(phase_img)
 
segmentator = FastSAMImageSegmentator()
segmentator.set_image(amplitude_img, 0)
masks = segmentator.segment(amplitude_img, 0)
