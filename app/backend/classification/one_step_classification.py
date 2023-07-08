from classification.classification import Classification
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import logging
import numpy as np


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
    
    def _prepare_data(self, X_updated, y_updated):
        num_sc = MinMaxScaler()
        pipe= Pipeline(steps=[("MinMaxScaler", num_sc)])
        num_feat = X_updated.columns
        ct = ColumnTransformer(transformers=[("NumericalTransformer", pipe, num_feat)])
        # Shuffle the data
        num_rows = X_updated.shape[0]
        permutation = np.random.permutation(num_rows)
        X_updated = X_updated.iloc[permutation].reset_index(drop=True)
        y_updated = y_updated[permutation]
        logging.info("Data preprocessed succesfully for retraining")

        return self._retrain_model(ct, X_updated, y_updated)
    
    def _retrain_model(self, ct, X_updated, y_updated):

        if self.model_type.lower() == 'svc':
            pipe_svc = Pipeline([
            ('ColumnTransformer', ct),
            ('classifier', SVC(max_iter=1000000, probability=True,
                               kernel = self.model.named_steps['classifier'].kernel,
                               C = self.model.named_steps['classifier'].C,
                               gamma = self.model.named_steps['classifier'].gamma))
            ])
            pipe_svc.fit(X_updated, y_updated)
            logging.info("Model SVC retrained succesfully")
            model_to_be_saved = pipe_svc

        elif self.model_type.lower() == 'rfc':
            pipe_rfc = Pipeline([
            ('ColumnTransformer', ct),
            ('classifier', RFC(random_state=12,
                               n_estimators = self.model.named_steps['classifier'].n_estimators,
                               max_features = self.model.named_steps['classifier'].max_features,
                               max_depth = self.model.named_steps['classifier'].max_depth))
            ])
            pipe_rfc.fit(X_updated, y_updated)
            logging.info("Model RFC retrained succesfully")
            model_to_be_saved = pipe_rfc

        elif self.model_type.lower() == 'knn':
            pipe_knn = Pipeline([
            ('ColumnTransformer', ct),
            ('classifier', KNeighborsClassifier(n_neighbors = self.model.named_steps['classifier'].n_neighbors,
                                                weights = self.model.named_steps['classifier'].weights,
                                                algorithm = self.model.named_steps['classifier'].algorithm))
            ])         
            pipe_knn.fit(X_updated, y_updated)
            logging.info("Model KNN retrained succesfully")
            model_to_be_saved = pipe_knn

        elif self.model_type.lower() == 'nb':
            pipe_nb = Pipeline([
                ('ColumnTransformer', ct),
                ('classifier', GaussianNB())
            ])
            pipe_nb.fit(X_updated, y_updated)
            logging.info("Model NB retrained succesfully")
            model_to_be_saved = pipe_nb
        
        return self._save_model(model_to_be_saved, self.models_folder, self.model_filename)