import numpy as np 
import os
import joblib

from abc import ABC, abstractmethod


class Classification(ABC):
    def __init__(self): 
        self.out_of_focus_label = "oof"
        self.aggregate_label = "agg"
        self.wbc_label = "wbc"
        self.plt_label = "plt"
        self.rbc_label = "rbc"

        self.labels = [self.out_of_focus_label, self.aggregate_label, self.wbc_label, self.plt_label, self.rbc_label]

        self.labels_column_name = "Labels"
        self.mask_id_column_name = "MaskID"

        self.models_folder = os.path.join("classification", "models")


    def classify(self, features):
        """
        We assume that 'features' is a pandas DataFrame whose columns are the different features extracted and 
        whose rows are the different cells found in the segmentation process.
        """
        features = self._drop_columns(features)

        predictions = self._get_predictions(features)
        probabilities = self._get_probabilities(features)
        return predictions, probabilities


    def _find_columns_to_drop(self, df):
        columns_to_drop = []
        if self.labels_column_name in df.columns:
            columns_to_drop.append(self.labels_column_name)
        if self.mask_id_column_name in df.columns:
            columns_to_drop.append(self.mask_id_column_name)
        return columns_to_drop


    def _drop_columns(self, features):
        """
        Removes unnecessary columns for classification, such as the "Mask ID" column or "Labels" column
        """
        features_copy = features.copy()
        columns_to_drop = self._find_columns_to_drop(features)
        
        if columns_to_drop:
            features_copy = features_copy.drop(columns_to_drop, axis=1)

        return features_copy
    
    def _load_model(self, folder_path, file_name):
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            return joblib.load(file_path)
        else:
            raise FileNotFoundError(f"Model file '{file_path}' does not exist.")


    @abstractmethod 
    def _get_predictions(self, features):
        pass
    

    @abstractmethod
    def _get_probabilities(self, features):
        pass

