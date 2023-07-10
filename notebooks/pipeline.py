from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from feature_extraction import * 
import numpy as np
from tqdm import tqdm
from cellpose_segmentation import CellposeSegmentation
from threshold_segmentation import ThresholdSegmentation
from sam_segmentation import SAMSegmentation
from cellpose import models
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import joblib


def load_model(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        print(f"Model file '{file_path}' does not exist.")
        return None
    

def image_to_masks(image):
    unique_labels = np.unique(image)  # Get the unique labels in the image

    masks = []
    for label in unique_labels:
        if label == 0:
            continue  # Skip the background label

        mask = np.where(image == label, 255, 0).astype(np.uint8)  # Convert to CV_8UC1 format
        masks.append(mask)

    return masks


class Segmentation(ABC):
    def __init__(self, phase_img, amplitude_img):
        # assert that there is the same number of phase and amplitude images
        assert(np.shape(amplitude_img)[0] == np.shape(phase_img)[0])

        self.batch = list(zip(phase_img, amplitude_img))
        self.masks = None
        self.ms = None


    def segment(self):
        self.masks = []
        self.ms = []
        for phase, amplitude in self.batch:
            mask_single_image = self._segment_single_image(phase, amplitude)
            self.masks.append(mask_single_image)
            self.ms.append(mask_single_image)

        return self.masks
    
    def outlines(self, object_mask):
        return self._list_of_outlines(object_mask)

    @abstractmethod
    def _segment_single_image(self, index, phase, amplitude):
        pass

    @abstractmethod
    def _list_of_outlines(self, object_mask):
        pass

class Classification:
    def __init__(self, X_test, model):
        # Classification initialization code here
        self.model = model
        self.X_test = X_test

    def classify(self):
        # Classification logic here
        y_pred = []
        labels = ['agg', 'oof', 'plt', 'rbc', 'wbc']
        prob = self.model.predict_proba(self.X_test)
        # Handle uncertain cases
        for i in range(self.X_test.shape[0]):
            max_prob = np.max(prob[i,:])
            second_max_prob = np.partition(prob[i,:], -2)[-2]

            if max_prob - second_max_prob > 0.15:
                y_pred.append(labels[np.argmax(prob[i,:])])
            else:
                y_pred.append("uncertain")

        return y_pred, prob




class Pipeline:
    def __init__(self, phase_img, amplitude_img, segmentation_algorithm, classification_model):
        self.phase_img = phase_img
        self.amplitude_img = amplitude_img
        self.segmentation_algorithm = segmentation_algorithm
        self.classification_model = classification_model
        self.seg = None
        self.masks_array = None

    def process_data(self):
        # Segmentation
        if self.segmentation_algorithm == 'cellpose':
            # model_type='cyto' or model_type='nuclei'
            seg_model = models.Cellpose(gpu=False, model_type='cyto')
            self.seg = CellposeSegmentation(self.phase_img, self.amplitude_img, seg_model)
        elif self.segmentation_algorithm == 'thresholding':
            self.seg = ThresholdSegmentation(self.phase_img, self.amplitude_img)
        elif self.segmentation_algorithm == 'sam':
            sam_checkpoint = "sam_vit_b_01ec64.pth"
            model_type = "vit_b"
            device = "cuda"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            seg_model = SamAutomaticMaskGenerator(sam)
            self.seg = SAMSegmentation(self.phase_img, self.amplitude_img, seg_model)
 
        masks = self.seg.segment()
        
        # probably here need an if statement
        self.masks_array = []
        for idx, _ in enumerate(self.phase_img):
            masks1 = image_to_masks(masks[idx])
            self.masks_array.append(masks1)

        # Feature extraction
        fe = FeatureExtractor(self.phase_img, self.amplitude_img, self.masks_array)
        extracted_features = fe.extract_features_multiple_masks()

        # Classification
        if self.classification_model == 'SVC':
            class_model = load_model("models", "best_svc_model.pkl")
        elif self.classification_model == 'RFC':
            class_model = load_model("models", "best_rfc_model.pkl")
        elif self.classification_model == 'KNN':
            class_model = load_model("models", "best_knn_model.pkl")
        elif self.classification_model == 'NB':
            class_model = load_model("models", "best_nb_model.pkl")
                
        classifier = Classification(extracted_features, class_model)

        y_pred, prob = classifier.classify()

        return y_pred, prob
    
    def get_outlines(self):
        outlines = []
        for ind, mask in enumerate(self.masks_array):
            outlines.append([])
            for object_mask in mask:
                outlines[ind].append(self.seg.outlines(object_mask))  
        return outlines
