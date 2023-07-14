import os
import joblib

import pandas as pd
import numpy as np

from classification.classification import Classification


class ThreeStepClassifier(Classification):
    def __init__(self, oof_model_filename=None, agg_model_filename=None, cell_model_filename=None):
        super().__init__()
        
        if oof_model_filename is None:
            self.oof_model_filename = "tsc_oof_model.pkl"
        else:
            self.oof_model_filename = oof_model_filename
        if agg_model_filename is None:
            self.agg_model_filename = "tsc_agg_model.pkl"
        else:
            self.agg_model_filename = agg_model_filename
        if cell_model_filename is None:
            self.cell_model_filename = "tsc_cell_model.pkl"
        else:
            self.cell_model_filename = cell_model_filename

        self.model_filename = {'oof': self.oof_model_filename, 'agg': self.agg_model_filename, 'cell': self.cell_model_filename}

        self.oof_model = self.load_model(self.models_folder, self.oof_model_filename)
        self.agg_model = self.load_model(self.models_folder, self.agg_model_filename)     
        self.cell_model = self.load_model(self.models_folder, self.cell_model_filename)     

        self.model = {'oof': self.oof_model, 'agg': self.agg_model, 'cell': self.cell_model}


    def save_model(self, folder_path, file_name):
        assert self.model is not None

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        oof_file_path = os.path.join(folder_path, 'oof_' + file_name)
        agg_file_path = os.path.join(folder_path, 'agg_' + file_name)
        cell_file_path = os.path.join(folder_path, 'cell_' + file_name)

        joblib.dump(self.oof_model, oof_file_path)
        joblib.dump(self.agg_model, agg_file_path)
        joblib.dump(self.cell_model, cell_file_path)


    def calculate_entropy(self, labels, probabilities):
        entropy = []
        for idx, label in enumerate(labels):
            if label == self.out_of_focus_label:
                probas = np.array(probabilities[idx]['oof_proba'])
            elif label == self.aggregate_label:
                probas = np.array(probabilities[idx]['agg_proba'])
            else:
                probas = np.array(probabilities[idx]['cell_proba'])
            entropy.append(-1 * np.sum(probas * np.log2(probas)))
        return entropy


    def _get_predictions(self, features):
        df = self._predict(features)
        return df[self.labels_column_name].tolist()

    
    def _get_probabilities(self, features):
        oof_proba, agg_proba, cell_proba = self._predict_proba(features)
        probabilities = []
        for idx in range(len(oof_proba)):
            probabilities.append({'oof_proba': oof_proba[idx].tolist(), 'agg_proba': agg_proba[idx].tolist(), 'cell_proba': cell_proba[idx].tolist(), 'proba': -1})

        return probabilities


    def _predict_proba(self, df):
        return self._predict_proba_oof(df), self._predict_proba_agg(df), self._predict_proba_cell(df)


    def _predict_proba_oof(self, df):
        return self.oof_model.predict_proba(df)


    def _predict_proba_agg(self, df):
        return self.agg_model.predict_proba(df)


    def _predict_proba_cell(self, df):
        return self.cell_model.predict_proba(df)


    def _predict(self, df):
        df_list = []

        oof_labels = self._oof_predict(df)
        df_in_focus = df[oof_labels == 0].copy()
        df_oof = df[oof_labels == 1].copy()
        df_oof[self.labels_column_name] = self.out_of_focus_label

        df_list.append(df_oof)
        
        agg_labels = self._agg_predict(df_in_focus)
        df_single_cell = df_in_focus[agg_labels == 0].copy()
        df_agg = df_in_focus[agg_labels == 1].copy()
        df_agg[self.labels_column_name] = self.aggregate_label
        df_list.append(df_agg)

        cell_labels = self._cell_predict(df_single_cell)

        df_single_cell.loc[:, self.labels_column_name] = cell_labels
        df_list.append(df_single_cell)

        return pd.concat(df_list).reindex(df.index)


    def _oof_predict(self, df):
        return self.oof_model.predict(df)


    def _agg_predict(self, df):
        return self.agg_model.predict(df)


    def _cell_predict(self, df): 
        return self.cell_model.predict(df)


    def name(self):
        return 'TSC'


    def fit(self, X, y, model_filename=None):
        """
        Method for retraining the models. Note that we use a "user_models_folder" to save them so we 
        do not overwrite our original models. 'model_filename' is an optional input for giving a 
        desired name to the model pickle file.
        """
        y_oof_binary = np.where(y == 'oof', 1, 0)
        y_agg_binary = np.where(y == 'agg', 1, 0)

        cell_mask = (y == 'plt') | (y == 'wbc') | (y == 'rbc')
        X_cells = X[cell_mask]
        y_cells = y[cell_mask]

        self.oof_model.fit(X, y_oof_binary) 
        self.agg_model.fit(X, y_agg_binary) 
        self.cell_model.fit(X_cells, y_cells) 

        self.save_model(self.user_models_folder, model_filename)
