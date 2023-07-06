import os
import joblib

import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, make_scorer, fbeta_score, precision_recall_curve, average_precision_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def save_model(model, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    joblib.dump(model, file_path)
    print(f"Model saved at {file_path}")


class NoTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X


def create_column_transformer(X, y):
# Calculate the min and max values for each feature and each class using the labeled data
    min_values = X.groupby(y).min()
    max_values = X.groupby(y).max()

# Define a transformer for each feature and each label that normalizes the feature
# with respect to the min and max values from the labeled data
    transformers = []
    for feature in X.columns:
        for label in y.unique():
            min_value = min_values[feature][label]
            max_value = max_values[feature][label]
            transformer = FunctionTransformer(lambda x: (x - min_value) / (max_value - min_value))
            transformers.append((f"Range_{feature}_{label}", transformer, [feature]))
        transformers.append((f"{feature}", NoTransformer(), [feature]))
      

# Use a ColumnTransformer to apply the transformers to the appropriate columns
    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return ct


def binary_logreg_gridsearch(preprocessing, X_train, y_train, metric, verbose=3):
    pipe = Pipeline([
            ('column_transform', preprocessing),
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest()),
            ('oversample', SMOTE(k_neighbors=3, random_state=42)),
            ('undersample', RandomUnderSampler(random_state=42)),
            ('classifier', LogisticRegression(max_iter=1000000, random_state=42, solver='saga'))
    ])


    param_grid = [
    {'column_transform': [None], 'scaler': [None, StandardScaler(), MinMaxScaler()], 'feature_selection__k': [5, 10, 15, 'all'], 
     'oversample': [SMOTE(k_neighbors=3, random_state=42)],
    'undersample': [RandomUnderSampler(random_state=42)],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10], 'classifier__class_weight': [None, 'balanced'],
     'classifier__penalty': ['l1', 'l2']},
    {'column_transform': [preprocessing], 'scaler': [None, StandardScaler(), MinMaxScaler()],
     'feature_selection__k': [5, 25, 50, 100, 'all'], 
     'oversample': [None],
    'undersample': [None], 'classifier__C': [0.001, 0.01, 0.1, 1, 10],
     'classifier__class_weight': [None, 'balanced'], 'classifier__penalty': ['l1', 'l2']},
        {'column_transform': [None], 'scaler': [None, StandardScaler(), MinMaxScaler()], 'feature_selection__k': [5, 10, 15, 'all'],
     'oversample': [None],
    'undersample': [None],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10], 'classifier__class_weight': [None, 'balanced'],
     'classifier__penalty': ['l1', 'l2']},
    {'column_transform': [preprocessing], 'scaler': [None, StandardScaler(), MinMaxScaler()],
     'feature_selection__k': [5, 25, 50, 100, 'all'], 
     'oversample': [SMOTE(k_neighbors=3, random_state=42)],
    'undersample': [RandomUnderSampler(random_state=42)],'classifier__C': [0.001, 0.01, 0.1, 1, 10],
     'classifier__class_weight': [None, 'balanced'], 'classifier__penalty': ['l1', 'l2']}
]

    #param_grid = list(ParameterGrid(gs_param))
    
    stratified = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=42)
    fhalfscorer = make_scorer(fbeta_score, beta=0.5)
    SCORING = {'ACC': 'accuracy', 'RE': 'recall', 'F1':'f1', 'AVG_PR': 'average_precision', 'BACC':'balanced_accuracy', 'FHALF': fhalfscorer}
    gs = GridSearchCV(pipe, param_grid, refit=metric, cv=stratified, scoring=SCORING, verbose=verbose, n_jobs=-1, return_train_score=True, error_score="raise")
            
    gs.fit(X_train, y_train)

    return gs

# Build LogisticReg Pipeline and return best estimator based on the grid search

def binary_xgboost_gridsearch(preprocessing, X_train, y_train, metric, verbose=3):
    pipe = Pipeline([
            ('column_transform', preprocessing),
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest()),
            ('oversample', SMOTE(k_neighbors=3, random_state=42)),
            ('undersample', RandomUnderSampler(random_state=42)),
            ('classifier', xgb.XGBClassifier(learning_rate=0.02, objective='binary:logistic', nthread=1, random_state=42))
    ])


    param_grid = [
    {'column_transform': [None], 'scaler': [None, StandardScaler(), MinMaxScaler()], 'feature_selection__k': [5, 10, 15, 'all'], 
     'oversample': [SMOTE(k_neighbors=3, random_state=42)],
    'undersample': [RandomUnderSampler(random_state=42)],
     'classifier__max_depth': [2, 3, 5, 7, 10],
     'classifier__n_estimators': [10, 100, 500],
    },
    {'column_transform': [preprocessing], 'scaler': [None, StandardScaler(), MinMaxScaler()],
     'feature_selection__k': [5, 25, 50, 100, 'all'], 
     'oversample': [None],
    'undersample': [None], 
     'classifier__max_depth': [2, 3, 5, 7, 10],
     'classifier__n_estimators': [10, 100, 500]},
        {'column_transform': [None], 'scaler': [None, StandardScaler(), MinMaxScaler()], 'feature_selection__k': [5, 10, 15, 'all'],
     'oversample': [None],
    'undersample': [None], 
     'classifier__max_depth': [2, 3, 5, 7, 10],
     'classifier__n_estimators': [10, 100, 500]},
    {'column_transform': [preprocessing], 'scaler': [None, StandardScaler(), MinMaxScaler()],
     'feature_selection__k': [5, 25, 50, 100, 'all'], 
     'oversample': [SMOTE(k_neighbors=3, random_state=42)],
    'undersample': [RandomUnderSampler(random_state=42)],
     'classifier__max_depth': [2, 3, 5, 7, 10],
     'classifier__n_estimators': [10, 100, 500]},
]
 
    stratified = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=42)

    fhalfscorer = make_scorer(fbeta_score, beta=0.5)
    SCORING = {'ACC': 'accuracy', 'RE': 'recall', 'F1':'f1', 'AVG_PR': 'average_precision', 'BACC':'balanced_accuracy', 'FHALF': fhalfscorer}
    gs = GridSearchCV(pipe, param_grid, refit=metric, cv=stratified, scoring=SCORING, verbose=verbose, n_jobs=-1, return_train_score=True, error_score="raise")
            
    gs.fit(X_train, y_train)

    return gs


def multiclass_logreg_gridsearch(preprocessing, X_train, y_train, metric, verbose=3):
    pipe = Pipeline([
            ('column_transform', preprocessing),
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest()),
            ('classifier', LogisticRegression(max_iter=1000000, random_state=42, solver='saga'))
    ])


    param_grid = [
    {'column_transform': [None], 'scaler': [None, StandardScaler(), MinMaxScaler()], 'feature_selection__k': [5, 10, 15, 'all'], 
     'classifier__C': [0.001, 0.01, 0.1, 1, 10], 'classifier__class_weight': [None, 'balanced'],
     'classifier__penalty': ['l1', 'l2']},
    {'column_transform': [preprocessing], 'scaler': [None, StandardScaler(), MinMaxScaler()],
     'feature_selection__k': [5, 25, 50, 100, 'all'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10],
     'classifier__class_weight': [None, 'balanced'], 'classifier__penalty': ['l1', 'l2']}
]
 
    stratified = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=42)
    fhalfscorer = make_scorer(fbeta_score, beta=0.5, average='macro')
    SCORING = {'ROC_OVO': 'roc_auc_ovo', 'F1_MICRO':'f1_micro', 'F1_MACRO':'f1_macro', 'ROC_OVR': 'roc_auc_ovr','FHALF_MACRO': fhalfscorer}
    gs = GridSearchCV(pipe, param_grid, refit=metric, cv=stratified, scoring=SCORING, verbose=verbose, n_jobs=-1, return_train_score=True, error_score="raise")
            
    gs.fit(X_train, y_train)

    return gs


def multiclass_xbgoost_gridsearch(preprocessing, X_train, y_train, metric, verbose=3):
    pipe = Pipeline([
            ('column_transform', preprocessing),
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest()),
            ('oversample', SMOTE(k_neighbors=3, random_state=42)),
            ('undersample', RandomUnderSampler(random_state=42)),
        ('classifier', xgb.XGBClassifier(learning_rate=0.02, objective='multi:softmax', nthread=1, random_state=42))
    ])


    param_grid = [
    {'column_transform': [None], 'scaler': [None, StandardScaler(), MinMaxScaler()], 'feature_selection__k': [5, 10, 15, 'all'], 
     'classifier__max_depth': [2, 3, 5, 7, 10],
     'classifier__n_estimators': [10, 100, 500],
    },
    {'column_transform': [preprocessing], 'scaler': [None, StandardScaler(), MinMaxScaler()],
     'feature_selection__k': [5, 25, 50, 100, 'all'], 
     'classifier__max_depth': [2, 3, 5, 7, 10],
     'classifier__n_estimators': [10, 100, 500]}]
  
    stratified = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=42)
    fhalfscorer = make_scorer(fbeta_score, beta=0.5, average='macro')
    SCORING = {'ROC_OVO': 'roc_auc_ovo', 'F1_MICRO':'f1_micro', 'F1_MACRO':'f1_macro', 'ROC_OVR': 'roc_auc_ovr','FHALF_MACRO': fhalfscorer}
    gs = GridSearchCV(pipe, param_grid, refit=metric, cv=stratified, scoring=SCORING, verbose=verbose, n_jobs=-1, return_train_score=True, error_score="raise")
            
    gs.fit(X_train, y_train)

    return gs


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_report_and_plot_confusion_matrices(y_train, y_test, y_train_pred, y_test_pred)
    print_balanced_accuracy(y_train, y_test, y_train_pred, y_test_pred)
    plot_precision_recall(model, X_train, y_train, title="PR Curve - Train Set")
    plot_precision_recall(model, X_test, y_test, title="PR Curve - Test Set")


def plot_precision_recall(model, X, y, title="Precision-Recall Curve"):
    yhat = model.predict_proba(X)
    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]
    # calculate roc curves
    precision, recall, thresholds = precision_recall_curve(y, yhat)
    # convert to f score
    f1score = (2 * precision * recall) / (precision + recall)
    pr_auc = average_precision_score(y, yhat)
    
    fhalfscore = ((5/4) * (precision * recall)) / (precision / 4 + recall)
    y_pred = np.zeros(len(y))
    
    # locate the index of the largest f score
    ix = np.argmax(f1score)
    
    y_pred[yhat > thresholds[ix]] = 1
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    print('Best Threshold=%f, F1-Score=%.3f, PR-AUC=%.3f, FHalf-Score=%.3f, BalancedAcc=%.3f' % (thresholds[ix], f1score[ix], pr_auc, fhalfscore[ix], balanced_accuracy))
    # plot the roc curve for the model
    no_skill = len(y[y==1]) / len(y)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.set_title(title)# show the plot
    plt.show()


def print_balanced_accuracy(y_train, y_test, y_train_pred, y_test_pred):
# Assuming y_pred and y_true are the predicted labels and ground truth, respectively
    weighted_accuracy = balanced_accuracy_score(y_train, y_train_pred)
    print("Balanced Accuracy (Train Set):", weighted_accuracy)
    weighted_accuracy_test = balanced_accuracy_score(y_test, y_test_pred)
    print("Balanced Accuracy (Test Set):", weighted_accuracy_test)


def print_report_and_plot_confusion_matrices(y_train, y_test, y_train_pred, y_test_pred):
        confusion_mtx_train = confusion_matrix(y_train, y_train_pred)
        classification_rep_train = classification_report(y_train, y_train_pred)

# Calculate confusion matrix and classification report
        confusion_mtx = confusion_matrix(y_test, y_test_pred)
        classification_rep = classification_report(y_test, y_test_pred)

# Print confusion matrix and classification report
        print("\nClassification Report Train Set:")
        print(classification_rep_train)
        print("\nClassification Report Test Set:")
        print(classification_rep)

# Create a figure and axes
        _, axs = plt.subplots(1, 2, figsize=(12, 5))
# Plot the confusion matrix heatmap
        sns.heatmap(confusion_mtx_train, annot=True, fmt="d", cmap="Blues", ax=axs[0])
        axs[0].set_title('Training Set')
# Plot the confusion matrix heatmap
        sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", ax=axs[1])
        axs[1].set_title('Test Set')

        for i in range(2):
# Set labels, title, and ticks
            axs[i].set_xlabel('Predicted Labels')
            axs[i].set_ylabel('True Labels')
            axs[i].set_title('Confusion Matrix')
            axs[i].xaxis.set_ticklabels(np.unique(y_train), rotation=90)
            axs[i].yaxis.set_ticklabels(np.unique(y_train), rotation=0)

# Show the plot
        plt.show()
