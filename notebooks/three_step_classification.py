import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


class ThreeStepClassifier:
    def __init__(self, oof_model, agg_model, cell_model,
                 range_labels_oof=['wbc', 'plt', 'agg', 'rbc'], 
                 range_labels_agg=['wbc', 'plt', 'rbc'],
                 range_labels_cell=['wbc', 'plt', 'rbc'],
                 num_features_oof=10,
                 num_features_agg=10,
                 num_features_cell=20,
                 agg_is_cell_type=False):
        self.oof_model = oof_model 
        self.agg_model = agg_model
        self.cell_model = cell_model

        self.oof_pipeline = Pipeline([
            ('oversample', SMOTE(random_state=42)),
            ('undersample', RandomUnderSampler(random_state=42)),
            ('classifier', self.oof_model)
        ])


        self.agg_pipeline = Pipeline([
            ('oversample', SMOTE(random_state=42)),
            ('undersample', RandomUnderSampler(random_state=42)),
            ('classifier', self.agg_model)
        ])

        self.cell_label_encoder = LabelEncoder()

        self.cell_pipeline = Pipeline([
            ('classifier', self.cell_model)
        ])

        self.num_features_oof = num_features_oof
        self.num_features_agg = num_features_agg
        self.num_features_cell = num_features_cell        

        self.range_labels_oof = range_labels_oof
        self.range_labels_agg = range_labels_agg
        self.range_labels_cell = range_labels_cell

        self.feature_ranges = None
        self.selected_features_oof = None
        self.selected_features_agg = None
        self.selected_features_cell = None

        self.out_of_focus_label = "oof"
        self.aggregate_label = "agg"
        self.wbc_label = "wbc"
        self.plt_label = "plt"
        self.rbc_label = "rbc"

        self.labels = [self.out_of_focus_label, self.aggregate_label, self.wbc_label, self.plt_label, self.rbc_label]


        self.labels_column_name = "Labels"
        self.mask_id_column_name = "Mask ID"
        self.image_id_column_name = "Image ID"
        
        # if agg_is_cell_type is True, then we consider agg as another type of cell
        self.agg_is_cell_type = agg_is_cell_type


    def fit(self, df):
        self.calculate_feature_ranges(df)
        df_in_focus = df[df[self.labels_column_name] != self.out_of_focus_label]
        df_single_cell = df_in_focus[df_in_focus[self.labels_column_name] != self.aggregate_label]
        self.oof_fit(df)
        if self.agg_is_cell_type:
            self.cell_fit(df_in_focus)
        else:
            self.agg_fit(df_in_focus)
            self.cell_fit(df_single_cell)


    def oof_fit(self, df):
        df_ranges = self.extract_feature_ranges(df, self.range_labels_oof)   

        X_init = df_ranges.drop(self.labels_column_name, axis=1)
        y = df_ranges[self.labels_column_name]
        y_binary = np.where(y == self.out_of_focus_label, 1, 0)

        best_features = self.extract_best_features(X_init, y_binary, self.num_features_oof)
        self.selected_features_oof = best_features

        X = df_ranges[best_features]
        self.oof_pipeline.fit(X, y_binary)


    def agg_fit(self, df):
        df_ranges = self.extract_feature_ranges(df, self.range_labels_agg)   

        X_init = df_ranges.drop(self.labels_column_name, axis=1)
        y = df_ranges[self.labels_column_name]
        y_binary = np.where(y == self.aggregate_label, 1, 0)

        best_features = self.extract_best_features(X_init, y_binary, self.num_features_agg)
        self.selected_features_agg = best_features

        X = df_ranges[best_features]

        self.agg_pipeline.fit(X, y_binary)


    def cell_fit(self, df): 
        df_ranges = self.extract_feature_ranges(df, self.range_labels_cell)   

        X_init = df_ranges.drop(self.labels_column_name, axis=1)
        # Create an instance of LabelEncoder

        # Fit and transform the 'Labels' column
        y_encoded = self.cell_label_encoder.fit_transform(df_ranges[self.labels_column_name])

        best_features = self.extract_best_features(X_init, y_encoded, self.num_features_cell)
        self.selected_features_cell = best_features

        X = df_ranges[best_features]

        self.cell_pipeline.fit(X, y_encoded)


    def extract_best_features(self, X, y, k):
        selector = SelectKBest(k=k)
        # Apply feaure selection to obtain the top k features
        selector.fit(X, y)

        # Get the indices of the selected featuresI
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = list(X.columns[selected_feature_indices])
        return selected_feature_names


    def calculate_feature_ranges(self, df): 
        feature_ranges = {}  # Dictionary to store typical ranges for each feature and each label
        features = list(df.drop(self.labels_column_name, axis=1).columns)
        for label in self.labels:
            label_data = df[df[self.labels_column_name] == label]
            for feature in features:
                feature_min = np.min(label_data[feature])
                feature_max = np.max(label_data[feature])
                feature_ranges[(feature, label)] = (feature_min, feature_max)
        self.feature_ranges = feature_ranges


    def find_columns_to_drop(self, df):
        columns_to_drop = []
        if self.labels_column_name in df.columns:
            columns_to_drop.append(self.labels_column_name)
        if self.mask_id_column_name in df.columns:
            columns_to_drop.append(self.mask_id_column_name)
        if self.image_id_column_name in df.columns:
            columns_to_drop.append(self.image_id_column_name)
        return columns_to_drop

        

    def extract_feature_ranges(self, df_original, labels):
        assert(self.feature_ranges is not None)

        df = df_original.copy()

        columns_to_drop = self.find_columns_to_drop(df_original)
        if columns_to_drop:
            features = list(df_original.drop(columns_to_drop, axis=1).columns)
        else:
            features = list(df_original.columns)

        new_columns = []
        for feature in features:
            for label in labels:
                feature_min, feature_max = self.feature_ranges[(feature, label)]
                new_column = (df[feature] - feature_min) / (feature_max - feature_min)
                new_column_name = f'{feature}Range{label}'
                new_columns.append(new_column.rename(new_column_name))
        df = pd.concat([df_original] + new_columns, axis=1)
        return df


    def predict_proba(self, df):
        return self.predict_proba_oof(df), self.predict_proba_agg(df), self.predict_proba_cell(df)


    def predict_proba_oof(self, df):
        df_ranges = self.extract_feature_ranges(df, self.range_labels_oof)
        return self.oof_model.predict_proba(df_ranges[self.selected_features_oof])


    def predict_proba_agg(self, df):
        df_ranges = self.extract_feature_ranges(df, self.range_labels_oof)
        return self.agg_model.predict_proba(df_ranges[self.selected_features_agg])


    def predict_proba_cell(self, df):
        df_ranges = self.extract_feature_ranges(df, self.range_labels_cell)
        return self.cell_model.predict_proba(df_ranges[self.selected_features_cell])


    def predict(self, df):
        df_list = []

        oof_labels = self.oof_predict(df)
        df_in_focus = df[oof_labels == 0].copy()
        df_oof = df[oof_labels == 1].copy()
        df_oof[self.labels_column_name] = self.out_of_focus_label

        df_list.append(df_oof)
        
        if self.agg_is_cell_type:
            df_single_cell = df_in_focus
        else:
            agg_labels = self.agg_predict(df_in_focus)
            df_single_cell = df_in_focus[agg_labels == 0].copy()
            df_agg = df_in_focus[agg_labels == 1].copy()
            df_agg[self.labels_column_name] = self.aggregate_label
            df_list.append(df_agg)

        cell_labels_encoded = self.cell_predict(df_single_cell)
        cell_labels = self.cell_label_encoder.inverse_transform(cell_labels_encoded)

        df_single_cell.loc[:, self.labels_column_name] = cell_labels
        df_list.append(df_single_cell)

        return pd.concat(df_list).reindex(df.index)

    def oof_predict(self, df):
        df_ranges = self.extract_feature_ranges(df, self.range_labels_oof)   
        return self.oof_pipeline.predict(df_ranges[self.selected_features_oof])

    def agg_predict(self, df):
        df_ranges = self.extract_feature_ranges(df, self.range_labels_agg)   
        return self.agg_pipeline.predict(df_ranges[self.selected_features_agg])

    def cell_predict(self, df): 
        df_ranges = self.extract_feature_ranges(df, self.range_labels_cell)   
        return self.cell_pipeline.predict(df_ranges[self.selected_features_cell])


