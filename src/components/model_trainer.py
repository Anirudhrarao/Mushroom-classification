import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    A class for training and evaluating machine learning models.

    Attributes:
        None

    Methods:
        __init__: Initialize the ModelTrainer object.
        initiate_model_trainer: Train and evaluate multiple machine learning models on given data.

    """

    def __init__(self) -> None:
        """
        Initialize the ModelTrainer object.

        Args:
            None

        Returns:
            None

        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_features_array, train_target_array, test_features_array, test_target_array):
        """
        Train and evaluate multiple machine learning models on given data.

        Args:
            train_features_array (array-like): Training data features.
            train_target_array (array-like): Training data targets.
            test_features_array (array-like): Test data features.
            test_target_array (array-like): Test data targets.

        Returns:
            float: Accuracy score of the best-performing model.
            sklearn.base.BaseEstimator: The best-performing model.

        Raises:
            CustomException: If an error occurs during model training and evaluation.

        """
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_features_array,
                train_target_array,
                test_features_array,
                test_target_array
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBoost Classifier": XGBClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "Ada Boost Classifier": AdaBoostClassifier(),
                "Support Vector Classifier": SVC(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False)
            }

            best_model = None
            best_accuracy_score = 0.0

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                if accuracy > best_accuracy_score:
                    best_accuracy_score = accuracy
                    best_model = model

            if best_accuracy_score < 0.6:
                raise CustomException("No Best Model Found", sys)

            logging.info("Best model found on both train and test dataset : {} with accuracy score : {}".format(
                best_model, best_accuracy_score))

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)

            logging.info("Using the best model found to predict on the test data")

            predicted = best_model.predict(X_test)

            Accuracy_Score = accuracy_score(y_test, predicted)
            logging.info("Prediction result on the test data : Accuracy Score -> {}".format(Accuracy_Score))

            return Accuracy_Score, best_model

        except Exception as e:
            raise CustomException(e, sys)
