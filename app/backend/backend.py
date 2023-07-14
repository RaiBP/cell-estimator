import h5py 
import cv2
import os

import pandas as pd
import numpy as np

from pathlib import Path

from feature_extraction.feature_extractor import FeatureExtractor
from image_loader import ImageLoader, prepare_phase_img, prepare_amplitude_img
from classification.utils import create_classification_model
from segmentation.utils import create_segmentation_model


class PipelineManager:
    def __init__(self,  logging, dataset_path: Path, segmentation_method: str, classification_method: str, feature_extractor: FeatureExtractor, user_defined_dataset_path):
        self.set_image_id(0)
        self.image_type = 0
        self.classification_method = classification_method
        self.cell_counter = 0
        self.image_counter = 0

        self.logging = logging

        self.set_dataset(dataset_path)
        self.set_segmentation_method(segmentation_method)
        self.set_classification_method(classification_method)
        self.set_feature_extractor(feature_extractor)
        self.shared_features = None
        self.predictions = None
        self.probabilities = None
        self.phase_image = None
        self.phase_image_str = None
        self.amplitude_image = None
        self.amplitude_image_str = None

        self.user_dataset = user_defined_dataset_path
        self._create_user_dataset()


    def set_dataset(self, dataset_path: Path):
        self.logging.info(f"Initializing image loader with new dataset with path {dataset_path}.")
        self.image_loader = ImageLoader.from_file(dataset_path)
        img_dims = self.image_loader.get_image_dimensions()
        self.img_dims = (img_dims[1], img_dims[2])
        self.dataset_id = dataset_path.name
        self.logging.info(f"Image loader initialized dataset with {len(self.image_loader)} images.")

    def set_segmentation_method(self, segmentation_method: str):
        self.logging.info(f"Initializing new segmentator of type {segmentation_method}.")
        self.image_segmentator = create_segmentation_model(segmentation_method)
        self.logging.info(f"New segmentator of type {segmentation_method} initialized.")

    def set_classification_method(self, classification_method: str):
        self.logging.info(f"Initializing new classifier of type {classification_method}.")
        self.classifier = create_classification_model(classification_method)
        self.logging.info(f"New classifier of type {classification_method} initialized.")

    def set_feature_extractor(self, feature_extractor: FeatureExtractor):
        self.logging.info(f"Initializing new feature extractor of type {feature_extractor}.")
        self.feature_extractor = feature_extractor
        self.logging.info(f"New feature extractor of type {feature_extractor} initialized.")

    def set_shared_features(self, shared_features: pd.DataFrame):
        self.shared_features = shared_features

    def get_shared_features(self):
        return self.shared_features

    def set_predictions(self, predictions):
        self.predictions = predictions

    def get_predictions(self):
        return self.predictions

    def get_current_segmentation_method(self) -> str:
        return self.image_segmentator.name()

    def get_current_classification_method(self) -> str:
        return self.classifier.name()

    def get_image_dimensions(self):
        return self.img_dims

    def set_dataset_id(self, dataset_id):
        self.dataset_id = dataset_id

    def set_image_id(self, image_id):
        self.image_id = image_id

    def get_dataset_id(self):
        return self.dataset_id

    def set_amplitude_phase_images(self, image_id):
        self.amplitude_image, self.phase_image = self.image_loader.get_images(image_id)
        self.amplitude_image_str = prepare_amplitude_img(self.amplitude_image)
        self.phase_image_str = prepare_phase_img(self.phase_image)

    def get_amplitude_phase_images(self):
        return self.amplitude_image, self.phase_image

    def get_amplitude_phase_images_str(self):
        return self.amplitude_image_str, self.phase_image_str

    def set_image_type(self, image_type):
        self.image_type = image_type


    def _create_user_dataset(self):
        # First run, we must create the h5 file
        with h5py.File(self.user_dataset, 'a') as f:
            # Check if the dataset_id group exists, if not, create it
            if self.dataset_id not in f:
                dataset_group = f.create_group(self.dataset_id)
                dataset_group.create_dataset('masks', shape=(0, *self.img_dims), maxshape=(None, *self.img_dims), dtype=np.uint8, chunks=(1, *self.img_dims))
                dataset_group.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
                dataset_group.create_dataset('image_ids', shape=(0,), maxshape=(None,), dtype=np.uint32)

    def get_saved_features(self, image_id, dataset_id, features_path):
        features = pd.read_csv(features_path)

        is_match_present = (features['DatasetID'] == dataset_id) & (features['ImageID'] == image_id)
        if any(is_match_present):
            # if we already have features by the image ID, we delete those
            return features[is_match_present]
        else:
            return None


    def get_saved_masks_and_labels(self, image_id, dataset_id): 
        with h5py.File(self.user_dataset, 'r') as f:
            if self.dataset_id not in f:
                return None, None

            dataset_group = f[dataset_id]
            image_ids_dataset = dataset_group['image_ids']

             # Check if the image_id already exists in the file
            if image_id not in image_ids_dataset:
                return None, None

            indeces = np.where(image_ids_dataset[:] == image_id)[0]
            mask_dataset = dataset_group['masks']
            label_dataset = dataset_group['labels']

            masks = mask_dataset[indeces]
            labels = label_dataset[indeces]

            return masks, labels

    def save_masks(self, masks, labels):
        # Save the masks and labels in the h5 file
        with h5py.File(self.user_dataset, 'a') as f:
            # Check if the dataset_id group exists, if not, create it
            if self.dataset_id not in f:
                self._create_user_dataset()

            dataset_group = f[self.dataset_id]
            
            image_ids_dataset = dataset_group['image_ids']

             # Check if the image_id already exists in the file
            if self.image_id in image_ids_dataset:

                start_occur = np.where(image_ids_dataset[:] == self.image_id)[0][0]
                end_occur = np.where(image_ids_dataset[:] == self.image_id)[0][-1]
                len_occur = (end_occur - start_occur) + 1
                len_masks = len(masks)
                init_len = dataset_group['masks'].shape[0]
                new_masks = len_occur - len_masks

                if len_masks == len_occur:
                    for mask, label in zip(masks, labels):
                        # Append the mask, label, and image_id to the corresponding datasets
                        mask_dataset = dataset_group['masks']
                        mask_dataset[start_occur] = mask

                        label_dataset = dataset_group['labels']
                        label_dataset[start_occur] = label

                        start_occur += 1

                elif len_masks > len_occur:
                    # Resize the arrays 
                    mask_dataset = dataset_group['masks']
                    mask_dataset.resize(init_len + (len_masks-len_occur), axis=0)

                    label_dataset = dataset_group['labels']
                    label_dataset.resize(init_len + (len_masks-len_occur), axis=0)

                    image_id_dataset = dataset_group['image_ids']
                    image_id_dataset.resize(init_len + (len_masks-len_occur), axis=0)

                    # Shift all the data after the Image-ID in question to the right
                    for k in range(init_len-(end_occur+1)):
                        mask_dataset[-1-k] = mask_dataset[init_len-1-k]
                        label_dataset[-1-k] = label_dataset[init_len-1-k]
                        image_id_dataset[-1-k] = image_id_dataset[init_len-1-k]

                    for mask, label in zip(masks, labels):
                        # Append the mask, label, and image_id to the corresponding datasets
                        mask_dataset[start_occur] = mask

                        label_dataset[start_occur] = label

                        image_id_dataset[start_occur] = self.image_id

                        start_occur += 1
                # len_masks < len_ocurr
                else:
                    mask_dataset = dataset_group['masks']
                    label_dataset = dataset_group['labels']
                    image_id_dataset = dataset_group['image_ids']

                    # Shift all the data after the Image-ID in question to the left
                    for k in range(init_len-(end_occur+1)):
                        mask_dataset[end_occur+(len_masks-len_occur)+1+k] = mask_dataset[end_occur+1+k]
                        label_dataset[end_occur+(len_masks-len_occur)+1+k] = label_dataset[end_occur+1+k]
                        image_id_dataset[end_occur+(len_masks-len_occur)+1+k] = image_id_dataset[end_occur+1+k]

                    for mask, label in zip(masks, labels):
                        # Append the mask, label, and image_id to the corresponding datasets
                        mask_dataset = dataset_group['masks']
                        mask_dataset[start_occur] = mask

                        label_dataset = dataset_group['labels']
                        label_dataset[start_occur] = label

                        image_id_dataset = dataset_group['image_ids']
                        image_id_dataset[start_occur] = self.image_id

                        start_occur += 1

                    # Resize the arrays
                    mask_dataset.resize(init_len + (len_masks-len_occur), axis=0)
                    label_dataset.resize(init_len + (len_masks-len_occur), axis=0)
                    image_id_dataset.resize(init_len + (len_masks-len_occur), axis=0)
                self.cell_counter += new_masks
            else:
                # Save the masks, labels, and image_ids as separate datasets
                for mask, label in zip(masks, labels):
                    # Append the mask, label, and image_id to the corresponding datasets
                    mask_dataset = dataset_group['masks']
                    mask_dataset.resize(mask_dataset.shape[0] + 1, axis=0)
                    mask_dataset[-1] = mask

                    label_dataset = dataset_group['labels']
                    label_dataset.resize(label_dataset.shape[0] + 1, axis=0)
                    label_dataset[-1] = label

                    image_id_dataset = dataset_group['image_ids']
                    image_id_dataset.resize(image_id_dataset.shape[0] + 1, axis=0)
                    image_id_dataset[-1] = self.image_id
 
                self.cell_counter += len(masks)
            # Increment the image_counter
            self.image_counter += 1


    def get_masks_from_polygons(self, polygons):
        """
        Method for converting the polygons to masks
        """
        image_shape = self.get_image_dimensions()
        masks = []
        # Process the received arrays
        for polygon in polygons:
            mask = np.zeros(image_shape, dtype=np.uint8)
            contour = np.array(polygon.points).reshape((-1, 2)) * np.array([image_shape[1], image_shape[0]])
            contour = contour.astype(np.int32)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            masks.append(mask)
        self.logging.info(f"{len(masks)} masks created successfully from received polygons")
        return masks
