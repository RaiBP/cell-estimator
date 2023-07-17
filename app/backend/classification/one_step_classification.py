import os
import glob
import joblib 
import numpy as np

from classification.classification import Classification


class OneStepClassifier(Classification):
    def __init__(self, model_type="SVC", model_filename=None, use_user_models=False):
        super().__init__(use_user_models=use_user_models)  
        self.model_type = model_type

        self.model_filename = self._get_model_filename() if model_filename is None else model_filename

        self.model = self.load_model(self.models_folder, self.model_filename)
        

    def save_model(self, folder_path, file_name, overwrite=False):
        assert self.model is not None
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)
        model_method = file_name.split("_")[1]
        if overwrite:
            self.delete_model(folder_path, model_method)
        joblib.dump(self.model, file_path)


    @staticmethod
    def delete_model(folder_path, model_method):
        files_to_delete = glob.glob(os.path.join(folder_path, f'*{model_method}*'))
        # Delete the files
        for file_path in files_to_delete:
            os.remove(file_path)


    def calculate_entropy(self, labels, probabilities):
        entropy = []
        for idx, _ in enumerate(labels):
            probas = np.array(probabilities[idx]['proba'])
            entropy.append(-1 * np.sum(probas * np.log2(probas)))
        return entropy

    def calculate_probability_per_label(self, labels, probabilities):
        probability = []
        classes = self.get_classes()
        for idx, _ in enumerate(labels):
            proba_dict = {}
            probas = probabilities[idx]['proba']
            for idx2, label in enumerate(classes):
                proba_dict[label] = probas[idx2]
            probability.append(proba_dict)
        return probability
            

    def get_classes(self):
        assert(self.model is not None)
        classes_enc = self.model.classes_
        classes = [item.decode('utf-8') for item in classes_enc]
        return classes


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
 
    
    def fit(self, X, y):
        X = self._drop_columns(X)
        self.model.fit(X, y) 
