import os 
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', "train.csv")
    test_data_path : str = os.path.join('artifacts', "test.csv")
    raw_data_path : str = os.path.join('artifacts', "data.csv")
    
    """
    Data ingestion configuration class.

    Args:
        train_data_path (str): The path to the train data file.
        test_data_path (str): The path to the test data file.
        raw_data_path (str): The path to the raw data file.
    """

    logging.info("Created data ingestion configuration")

class DataIngestion:
    def __init__(self):
        """
        Data ingestion class.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiate data ingestion process.

        Returns:
            (str, str): The paths to the train and test data files.
        """
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv("dataset/mushroom_data.csv")
            logging.info("Read the raw dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split is now initiated")

            train_set, test_set = train_test_split(df, test_size=0.25, random_state=222)

            logging.info("Train test split is now completed")
            logging.info("Saving train set and test set as csv files")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is now completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_features, train_target, test_features, test_target,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_features, train_target, test_features, test_target))
