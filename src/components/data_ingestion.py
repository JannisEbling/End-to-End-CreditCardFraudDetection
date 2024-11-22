import os
import sys

import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception.exception import CreditCardException
from src.logging.logger import logging

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
        except Exception as e:
            raise CreditCardException(e, sys) from e

    def export_collection_as_dataframe(self):
        try:
            # First try MongoDB
            try:
                # Get the collection
                database_name = self.data_ingestion_config.database_name
                collection_name = self.data_ingestion_config.collection_name
                collection = self.mongo_client[database_name][collection_name]

                logging.info("Requesting data from MongoDB Database")

                # Get all data from collection and drop the id column
                df = pd.DataFrame(list(collection.find()))
                if "_id" in df.columns.to_list():
                    df = df.drop(columns=["_id"], axis=1)

                # Replace MongoDBS "na" with numpys np.nan
                df.replace({"na": np.nan}, inplace=True)

                # Limit to 10,000 cases
                if len(df) > 10000:
                    df = df.sample(n=10000, random_state=42)

                logging.info("Successfully retrieved data from MongoDB")
                return df

            except Exception as mongo_error:
                logging.warning(f"Failed to get data from MongoDB: {str(mongo_error)}")
                logging.info("Falling back to local data file")

                # Fallback to data/creditcard_2023.csv
                file_path = os.path.join("data", "creditcard_2023.csv")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Fallback data file not found at {file_path}"
                    )

                df = pd.read_csv(file_path)
                df.replace({"na": np.nan}, inplace=True)

                # Limit to 10,000 cases
                if len(df) > 10000:
                    df = df.sample(n=10000, random_state=42)

                logging.info("Successfully retrieved data from local file")
                return df

        except Exception as e:
            raise CreditCardException(e, sys) from e

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            # Create folder to save data
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            # Save dataframe
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            logging.info("Saved data from the Database")

            return dataframe

        except Exception as e:
            raise CreditCardException(e, sys) from e

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            # Perform train test split
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            # Create folder to save split data
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path.")
            # Save train and test data
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info("Exported train and test file path.")

        except Exception as e:
            raise CreditCardException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return dataingestionartifact

        except Exception as e:
            raise CreditCardException(e, sys) from e