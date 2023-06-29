from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import numpy as np



class Segmentation(ABC):
    def __init__(self, phase_array, amplitude_array):
        # assert that there is the same number of phase and amplitude images
        assert(np.shape(amplitude_array)[0] == np.shape(phase_array)[0])

        self.batch = list(zip(phase_array, amplitude_array))
        self.masks = None


    def segment(self):
        self.masks = []
        for phase, amplitude in self.batch:
            mask_single_image = self._segment_single_image(phase, amplitude)
            self.masks.append(mask_single_image)
        return self.masks
    
    def outlines(self, mask):
        return self._list_of_outlines(mask)

    @abstractmethod
    def _segment_single_image(self, index, phase, amplitude):
        pass

    @abstractmethod
    def _list_of_outlines(self, mask):
        pass




class FeatureExtraction:
    def __init__(self):
        # Feature extraction initialization code here
        pass

    def extract_features(self, data):
        # Feature extraction logic here
        pass






class Classification:
    def __init__(self, X_test, model, prob = True):
        # Classification initialization code here
        self.model = model
        self.X_test = X_test
        self.prob = prob

    def classify(self):
        # Classification logic here
        if self.prob == True:
            y_pred = self.model.predict_proba(self.X_test)
        else:
            y_pred = self.model.predict(self.X_test)
        return y_pred




class Pipeline:
    def __init__(self, segmentation, fetaure_extraction, classification):
        self.segmentation = segmentation
        self.feature_extraction = fetaure_extraction
        self.classification = classification

    def process_data(self, data):
        masks = self.segmentation.segment()
        extracted_features = self.feature_extraction.extract_features()
        classification_result = self.classification.classify()
        return classification_result
