import h5py 

import pandas as pd
import numpy as np

from pathlib import Path

from feature_extraction.feature_extractor import FeatureExtractor
from image_loader import ImageLoader
from classification.utils import create_classification_model
from segmentation.utils import create_segmentation_model


class PipelineManager:
    def __init__(self,  logging, dataset_path: Path, segmentation_method: str, classification_method: str, feature_extractor: FeatureExtractor, user_defined_dataset_path):
        self.image_id = 0
        self.classification_method = classification_method
        self.cell_counter = 0
        self.image_counter = 0
        self.user_dataset = user_defined_dataset_path

        self._create_user_dataset()

        self.set_dataset(dataset_path)
        self.set_segmentation_method(segmentation_method)
        self.set_classification_method(classification_method)
        self.set_feature_extractor(feature_extractor)
        self.shared_features = None
        self.predictions = None

        self.logging = logging


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

    def set_predictions(self, predictions):
        self.predictions = predictions

    def get_predictions(self):
        return self.predictions

    def get_current_segmentation_method(self) -> str:
        return self.image_segmentator.name()

    def set_dataset_id(self, dataset_id):
        self.dataset_id = dataset_id


    def set_image_id(self, image_id):
        self.image_id = image_id


    def _create_user_dataset(self):
        # First run, we must create the h5 file
        with h5py.File(self.user_dataset, 'a') as f:
            # Check if the dataset_id group exists, if not, create it
            if self.dataset_id not in f:
                dataset_group = f.create_group(self.dataset_id)
                dataset_group.create_dataset('masks', shape=(0, *self.img_dims), maxshape=(None, *self.img_dims), dtype=np.uint8)
                dataset_group.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
                dataset_group.create_dataset('image_ids', shape=(0,), maxshape=(None,), dtype=np.uint32)


    def save_masks(self, masks, labels):
        # Save the masks and labels in the h5 file
        with h5py.File(self.user_dataset, 'a') as f:
            # Check if the dataset_id group exists, if not, create it
            if self.dataset_id not in f:
                self._create_user_dataset()

            dataset_group = f[self.dataset_id]

            # Check if the image_id already exists in the file
            image_ids_dataset = dataset_group['image_ids']
            if self.image_id in image_ids_dataset:
                # Overwrite the existing mask and label data
                index = np.where(image_ids_dataset[:] == self.image_id)[0][0]
                masks_dataset = dataset_group['masks']
                masks_dataset[index] = masks
                labels_dataset = dataset_group['labels']
                labels_dataset[index] = labels
            else:
                # Append the new mask, label, and image_id data
                image_ids_dataset.resize(image_ids_dataset.shape[0] + 1, axis=0)
                image_ids_dataset[-1] = self.image_id
                masks_dataset = dataset_group['masks']
                masks_dataset.resize(masks_dataset.shape[0] + 1, axis=0)
                masks_dataset[-1] = masks
                labels_dataset = dataset_group['labels']
                labels_dataset.resize(labels_dataset.shape[0] + 1, axis=0)
                labels_dataset[-1] = labels

            # Increment the image_counter
            self.image_counter += 1
            self.cell_counter += len(masks)
