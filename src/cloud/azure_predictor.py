import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
from src.constant.training_pipeline import (
    AZURE_ML_WORKSPACE,
    AZURE_ML_SUBSCRIPTION_ID,
    AZURE_ML_RESOURCE_GROUP,
    AZURE_ML_ENDPOINT_NAME,
)
from src.exception.exception import CreditCardException
from src.logging.logger import logging
import sys


class AzurePredictor:
    def __init__(self):
        try:
            # Initialize Azure ML client
            self.credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=AZURE_ML_SUBSCRIPTION_ID,
                resource_group_name=AZURE_ML_RESOURCE_GROUP,
                workspace_name=AZURE_ML_WORKSPACE,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, data: pd.DataFrame) -> list:
        """
        Make predictions using Azure ML endpoint
        Returns predictions if successful, None if fails
        """
        try:
            # Get endpoint
            endpoint = self.ml_client.online_endpoints.get(name=AZURE_ML_ENDPOINT_NAME)

            # Convert data to dictionary format expected by the endpoint
            data_dict = data.to_dict(orient="records")

            # Make prediction
            response = self.ml_client.online_endpoints.invoke(
                endpoint_name=AZURE_ML_ENDPOINT_NAME, request_file=data_dict
            )

            return response

        except Exception as e:
            logging.error(f"Azure prediction failed: {str(e)}")
            return None  # Return None to indicate failure and trigger fallback
