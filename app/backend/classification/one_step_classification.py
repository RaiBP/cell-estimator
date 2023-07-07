from classification.classification import Classification


class OneStepClassifier(Classification):
    def __init__(self, model_type="SVC", model_filename=None):
        super().__init__()  
        self.model_type = model_type


        if model_filename is None:
            self.model_filename = self._get_model_filename()
        else:
            self.model_filename = model_filename
        
        self.model = self._get_model()


    def _get_probabilities(self, features):
        assert(self.model is not None)
        probas = self.model.predict_proba(features)
        list_of_dicts = [{'proba': prob_list.tolist()} for prob_list in probas]
        return list_of_dicts


    def _get_predictions(self, features):
        assert(self.model is not None)
        binary_list = self.model.predict(features)

        return [item.decode('utf-8') for item in binary_list]

    
    def _get_model_filename(self):
        return "best_" + self.model_type.lower() + "_model.pkl"


    def _get_model(self):
        return self._load_model(self.models_folder, self.model_filename)
