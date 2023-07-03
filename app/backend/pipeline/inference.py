from feature_extraction.feature_extractor import FeatureExtractor
from segmentation.segmentation import Segmentation
from classification.classification import Classification


class Inference:
    def __init__(self, segmentation_method, classification_method):
        self.segmentation_method = segmentation_method
        self.classification_method = classification_method

        self.segmentation_model = Segmentation.create_model(self.segmentation_method)
        self.feature_extractor = FeatureExtractor()
        self.classification_model = Classification.create_model(self.classification_method)


    def _segment(self, phase, amplitude):
         return self.segmentation_model.segment(phase, amplitude)    


    def _extract_features(self, phase, amplitude, masks):
        return self.feature_extractor.extract_features(phase, amplitude, masks)

    def _classify(self, features):
        return self.classification_model.classify(features)


    def run(self, phase, amplitude):
        # Segmentation
        masks = self._segment(phase, amplitude)

        # Feature extraction
        extracted_features = self._extract_features(phase, amplitude, masks)

        # Classification
        predictions, probabilities = self._classify(extracted_features) 

        return predictions, probabilities, masks, extracted_features
