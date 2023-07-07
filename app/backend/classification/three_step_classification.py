import pandas as pd

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

        self.oof_model = self._load_model(self.models_folder, self.oof_model_filename)
        self.agg_model = self._load_model(self.models_folder, self.agg_model_filename)     
        self.cell_model = self._load_model(self.models_folder, self.cell_model_filename)     


    def _get_predictions(self, features):
        df = self._predict(features)
        return df[self.labels_column_name].tolist()

    
    def _get_probabilities(self, features):
        oof_proba, agg_proba, cell_proba = self._predict_proba(features)
        probabilities = []
        for idx in range(len(oof_proba)):
            probabilities.append({'oof_proba': oof_proba[idx].tolist(), 'agg_proba': agg_proba[idx].tolist(), 'cell_proba': cell_proba[idx].tolist()})

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


