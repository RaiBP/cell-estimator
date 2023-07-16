from classification.one_step_classification import OneStepClassifier
from classification.three_step_classification import ThreeStepClassifier


def create_classification_model(selector):
    base_strings = ['tsc', '3sc', '3 step classifier', '3 step classification', '3 step']
    similar_strings = []
    for base_string in base_strings:
        # Generate similar strings by replacing spaces with hyphens
        similar_strings.append(base_string.replace(' ', '-'))
        
        # Generate similar strings by removing hyphens
        similar_strings.append(base_string.replace(' ', ''))
    
        # Generate similar strings by removing hyphens
        similar_strings.append(base_string.replace('3', 'three'))

    similar_strings += base_strings
        
    if selector.lower() in similar_strings:
        return ThreeStepClassifier()
    else:
        return OneStepClassifier(selector)


def list_classification_methods() -> list[str]:
    return ['svc', 'knn', 'rfc', 'tsc']