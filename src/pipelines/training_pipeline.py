from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_features, train_target, test_features, test_target,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_features, train_target, test_features, test_target))
