import os
import joblib 
import numpy as np

from classification.classification import Classification


class OneStepClassifier(Classification):
    def __init__(self, model_type="SVC", model_filename=None):
        super().__init__()  
        self.model_type = model_type

        self.model_filename = self._get_model_filename() if model_filename is None else model_filename

        self.model = self.load_model(self.models_folder, self.model_filename)
        

    def save_model(self, folder_path, file_name):
        assert self.model is not None
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        joblib.dump(self.model, file_path)


    def calculate_entropy(self, labels, probabilities):
        entropy = []
        for idx, _ in enumerate(labels):
            probas = np.array(probabilities[idx]['proba'])
            entropy.append(-1 * np.sum(probas * np.log2(probas)))
        return entropy


    def _get_probabilities(self, features):
        assert(self.model is not None)
        probas = self.model.predict_proba(features)
        list_of_dicts = [{'oof_proba': -1, 'agg_proba': -1, 'cell_proba': -1, 'proba': prob_list.tolist()} for prob_list in probas]
        return list_of_dicts


    def _get_predictions(self, features):
        assert(self.model is not None)
        binary_list = self.model.predict(features)

        return [item.decode('utf-8') for item in binary_list]

    
    def _get_model_filename(self):
        return "osc_" + self.model_type.lower() + "_model.pkl"

    def name(self):
        return self.model_type
 
    
    def fit(self, X, y, model_filename=None):
        X = self._drop_columns(X)
        self.model.fit(X, y) 

        model_filename = self.model_filename if model_filename is None else model_filename

        self.save_model(self.user_models_folder, model_filename)
