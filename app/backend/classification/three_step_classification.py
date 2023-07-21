import os
import glob
import joblib

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from classification.classification import Classification


class ThreeStepClassifier(Classification):
    def __init__(self, oof_model_filename=None, agg_model_filename=None, cell_model_filename=None, use_user_models=False):
        super().__init__(use_user_models=use_user_models)

        self.oof_model_filename = self._get_model_filename("oof", use_user_models, oof_model_filename)
        self.agg_model_filename = self._get_model_filename("agg", use_user_models, agg_model_filename)
        self.cell_model_filename = self._get_model_filename("cell", use_user_models, cell_model_filename)

        self.model_filename = {'oof': self.oof_model_filename, 'agg': self.agg_model_filename, 'cell': self.cell_model_filename}

        oof_model = self.load_model(self.models_folder, self.oof_model_filename)
        self.oof_model = oof_model[0]
        self.oof_threshold = oof_model[1]
        agg_model = self.load_model(self.models_folder, self.agg_model_filename)
        self.agg_model = agg_model[0]
        self.agg_threshold = agg_model[1]
        self.cell_model = self.load_model(self.models_folder, self.cell_model_filename)     

        self.model = {'oof': self.oof_model, 'agg': self.agg_model, 'cell': self.cell_model}


    def _get_model_filename(self, step, use_user_models, given_filename):
        if given_filename is not None:
            return given_filename 
        if use_user_models: 
            model_filename = Path(glob.glob(os.path.join(self.models_folder, f'*{step}*'))[0]).name
            number = model_filename.split("_")[2]
            return "tsc_" + step + "_" + number + "_model.pkl"
        return "tsc_" + step + "_model.pkl"


    def set_model_filename(self, oof_model_filename, agg_model_filename, cell_model_filename):
        self.oof_model_filename = oof_model_filename
        self.agg_model_filename = agg_model_filename
        self.cell_model_filename = cell_model_filename


    def save_model(self, folder_path, file_name, overwrite=False):
        assert self.model is not None

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        oof_filename = 'tsc_oof_' + file_name
        agg_filename = 'tsc_agg_' + file_name
        cell_filename = 'tsc_cell_' + file_name

        oof_file_path = os.path.join(folder_path, oof_filename)
        agg_file_path = os.path.join(folder_path, agg_filename)
        cell_file_path = os.path.join(folder_path, cell_filename)

        if overwrite:
            self.delete_model(folder_path)

        joblib.dump((self.oof_model, self.oof_threshold), oof_file_path)
        joblib.dump((self.agg_model, self.agg_threshold), agg_file_path)
        joblib.dump(self.cell_model, cell_file_path)

        self.set_model_filename(oof_filename, agg_filename, cell_filename)
    

    @staticmethod
    def delete_model(folder_path):
        labels = ['oof', 'agg', 'cell']
        # Find all files in the folder that match the string
        for label in labels:
            files_to_delete = glob.glob(os.path.join(folder_path, f'*{label}*'))
            # Delete the files
            for file_path in files_to_delete:
                os.remove(file_path)


    def calculate_entropy(self, labels, probabilities):
        entropy = []
        for idx, label in enumerate(labels):
            if label == self.out_of_focus_label:
                probas = np.array(probabilities[idx]['oof_proba'])
            elif label == self.aggregate_label:
                probas = np.array(probabilities[idx]['agg_proba'])
            else:
                probas = np.array(probabilities[idx]['cell_proba'])

            non_zero_probas = probas[probas != 0]
            entropy.append(-1 * np.sum(non_zero_probas * np.log2(non_zero_probas)))
        return entropy

    def get_classes(self):
        oof_binary_classes = self.oof_model.classes_
        oof_classes = ['infocus' if value == 0 else 'oof' for value in oof_binary_classes]

        agg_binary_classes = self.agg_model.classes_
        agg_classes = ['single_cell' if value == 0 else 'agg' for value in agg_binary_classes]
        
        return {'oof': oof_classes, 'agg': agg_classes, 'cell': self.cell_model.classes_}

    def calculate_probability_per_label(self, labels, probabilities):
        probability = []
        classes = self.get_classes()
        oof_classes = list(classes['oof'])
        agg_classes = list(classes['agg'])
        cell_classes = list(classes['cell'])
        for idx, _ in enumerate(labels):
            proba_dict = {}
            probas_oof = probabilities[idx]['oof_proba']
            probas_agg = probabilities[idx]['agg_proba']
            probas_cell = probabilities[idx]['cell_proba'] 

            prob_oof = probas_oof[oof_classes.index('oof')]
            prob_agg = probas_agg[agg_classes.index('agg')]
            prob_wbc = probas_cell[cell_classes.index('wbc')]
            prob_rbc = probas_cell[cell_classes.index('rbc')]
            prob_plt = probas_cell[cell_classes.index('plt')]

            total_prob_agg = (1-prob_oof) * prob_agg
            total_prob_wbc = (1-prob_oof) * (1-prob_agg) * prob_wbc
            total_prob_rbc = (1-prob_oof) * (1-prob_agg) * prob_rbc
            total_prob_plt = (1-prob_oof) * (1-prob_agg) * prob_plt

            proba_dict['oof'] = prob_oof 
            proba_dict['agg'] = total_prob_agg
            proba_dict['wbc'] = total_prob_wbc
            proba_dict['rbc'] = total_prob_rbc
            proba_dict['plt'] = total_prob_plt

            probability.append(proba_dict)
        return probability

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
        if df.empty:
            return pd.DataFrame(columns=df.columns)

        df_list = []

        oof_labels = self._oof_predict(df)
        df_in_focus = df[oof_labels == 0].copy()
        df_oof = df[oof_labels == 1].copy()
        df_oof[self.labels_column_name] = self.out_of_focus_label

        df_list.append(df_oof)

        if df_in_focus.empty:
            return df_oof.reindex(df.index)
        
        agg_labels = self._agg_predict(df_in_focus)
        df_single_cell = df_in_focus[agg_labels == 0].copy()
        df_agg = df_in_focus[agg_labels == 1].copy()
        df_agg[self.labels_column_name] = self.aggregate_label
        df_list.append(df_agg)

        if not df_single_cell.empty:
            cell_labels = self._cell_predict(df_single_cell)

            df_single_cell.loc[:, self.labels_column_name] = cell_labels
            df_list.append(df_single_cell)

        return pd.concat(df_list).reindex(df.index)


    def _oof_predict(self, df):
        y_pred_proba = self.oof_model.predict_proba(df)
        y_pred_mask = y_pred_proba[:, 1] > self.oof_threshold
        y_pred = np.zeros(len(y_pred_proba))
        y_pred[y_pred_mask] = 1
        return y_pred


    def _agg_predict(self, df):
        y_pred_proba = self.agg_model.predict_proba(df)
        y_pred_mask = y_pred_proba[:, 1] > self.agg_threshold
        y_pred = np.zeros(len(y_pred_proba))
        y_pred[y_pred_mask] = 1
        return y_pred


    def _cell_predict(self, df): 
        return self.cell_model.predict(df)


    def name(self):
        return 'TSC'

    @staticmethod
    def calculate_optimal_threshold(model, X_df, y, n_splits=5, n_repeats=10, random_state=43):
        X = X_df.values
        rsfkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        optimal_thresholds = []
        for train_index, test_index in rsfkf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            yhat = model.predict_proba(X_test)[:, 1]

            precision, recall, thresholds = precision_recall_curve(y_test, yhat)
            sum_pr = precision + recall
            if np.any(np.array(sum_pr) == 0):
                continue
            f1score = (2 * precision * recall) / sum_pr

            # locate the index of the largest f score
            ix = np.argmax(f1score) 
                        
            optimal_thresholds.append(thresholds[ix]) 

        return np.median(optimal_thresholds)

    def fit(self, X, y):
        """
        Method for retraining the models.
        """
        X = self._drop_columns(X)

        y_oof_binary = np.where(y == 'oof', 1, 0)
        y_agg_binary = np.where(y == 'agg', 1, 0)

        cell_mask = (y == 'plt') | (y == 'wbc') | (y == 'rbc')
        X_cells = X[cell_mask]
        y_cells = y[cell_mask]


        self.oof_threshold = self.calculate_optimal_threshold(self.oof_model, X, y_oof_binary, n_splits=2, n_repeats=20)
        self.agg_threshold = self.calculate_optimal_threshold(self.agg_model, X, y_agg_binary, n_splits=2, n_repeats=20)

        self.oof_model.fit(X, y_oof_binary) 
        self.agg_model.fit(X, y_agg_binary) 
        
        self.cell_model.fit(X_cells, y_cells) 

