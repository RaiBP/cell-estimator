import h5py 
import numpy as np


class PipelineManager:
    def __init__(self, dataset, segmentation_method, classification_method, user_defined_dataset_path, img_dims):
        self.image_id = 0
        self.dataset_id = dataset
        self.segmentation_method = segmentation_method
        self.classification_method = classification_method
        self.cell_counter = 0
        self.image_counter = 0
        self.user_dataset = user_defined_dataset_path
        self.img_dims = (img_dims[1], img_dims[2])

        self._create_user_dataset()

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
