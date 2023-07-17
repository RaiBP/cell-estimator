import os
from pathlib import Path 
import pandas as pd
from classification.one_step_classification import OneStepClassifier
from classification.three_step_classification import ThreeStepClassifier


def create_classification_model(selector, use_user_trained_models):
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
        return ThreeStepClassifier(use_user_models=use_user_trained_models)
    else:
        return OneStepClassifier(selector, use_user_models=use_user_trained_models)

def list_files(path):
    files = []
    for file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_name)):
            files.append(file_name)
    return files

def list_classification_methods(user_models_folder) -> list[str]:    
    user_models = list_files(user_models_folder)
    model_list =  ['SVC', 'KNN', 'RFC', 'Three Step Classification']

    if len(user_models) != 0:
        for model in user_models:
            separated_name = model.split("_")
            model_type = separated_name[0] # either osc or tsc
            model_method = separated_name[1] # either knn, nb, rfc, svc, agg, cell or oof
            model_examples = separated_name[2] # number of training examples used

            if model_type == 'osc':
                model_list.append(f"{model_method.upper()} (retrained with {model_examples} examples)")
            elif model_type == 'tsc':
                model_list.append(f"Three Step Classification (retrained with {model_examples} examples)")
            else:
                continue
    return model_list
