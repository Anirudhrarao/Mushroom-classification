import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def save_object(file_path, obj):
    """
    Save an object to a file.

    Args:
        file_path (str): The path to the file.
        obj: The object to be saved.

    Returns:
        None.
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

    print(f"The object is saved to {file_path}")

def evaluate_model(TrainFeatures, TrainTarget, TestFeatures, TestTarget, models, params):
    """
    Evaluate a list of models on a dataset.

    Args:
        TrainFeatures (pd.DataFrame): The training features.
        TrainTarget (pd.Series): The training target.
        TestFeatures (pd.DataFrame): The test features.
        TestTarget (pd.Series): The test target.
        models (dict): A dictionary of models, where the keys are the model names and the values are the models.
        params (dict): A dictionary of hyperparameters, where the keys are the model names and the values are the hyperparameters.

    Returns:
        A dictionary of evaluation results, where the keys are the model names and the values are the evaluation results.
    """

    try:
        report = {}

        for model_name, model in models.items():
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(TrainFeatures, TrainTarget)

            model.set_params(**gs.best_params_)
            model.fit(TrainFeatures, TrainTarget)

            y_train_pred = model.predict(TrainFeatures)
            y_test_pred = model.predict(TestFeatures)

            train_model_score = accuracy_score(TrainTarget, y_train_pred)
            test_model_score = accuracy_score(TestTarget, y_test_pred)

            report[model_name] = {
                "train_score": train_model_score,
                "test_score": test_model_score,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        The object.
    """

    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

