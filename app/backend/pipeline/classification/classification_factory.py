from classification.one_step_classification import OneStepClassifier
from classification.three_step_classification import ThreeStepClassifier


class ClassificationFactory:
    def __init__(self):
        pass 


    @staticmethod
    def create_model(selector):
        if selector.lower() == 'tsc':
            return ThreeStepClassifier()
        else:
            return OneStepClassifier(selector)

