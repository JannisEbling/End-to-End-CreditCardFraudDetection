import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    Model,
)
from azure.identity import DefaultAzureCredential
from src.constant.training_pipeline import (
    AZURE_ML_WORKSPACE,
    AZURE_ML_SUBSCRIPTION_ID,
    AZURE_ML_RESOURCE_GROUP,
    AZURE_ML_ENDPOINT_NAME,
    AZURE_ML_DEPLOYMENT_NAME,
    AZURE_ML_MODEL_NAME,
    AZURE_ML_ENVIRONMENT_NAME,
    AZURE_ML_INSTANCE_TYPE,
    AZURE_ML_INSTANCE_COUNT,
)
from src.exception.exception import CreditCardException
from src.logging.logger import logging
import sys


class AzureMLSetup:
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
            raise CreditCardException(e, sys)

    def create_environment(self):
        try:
            # Create environment with required packages
            env = Environment(
                name=AZURE_ML_ENVIRONMENT_NAME,
                description="Environment for CreditCard  model",
                conda_file="environment.yml",  # You'll need to create this
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            )
            self.ml_client.environments.create_or_update(env)
            return env
        except Exception as e:
            raise CreditCardException(e, sys)

    def register_model(self, model_path: str):
        try:
            # Register the model
            model = Model(
                path=model_path,
                name=AZURE_ML_MODEL_NAME,
                description="CreditCard  Detection Model",
            )
            self.ml_client.models.create_or_update(model)
            return model
        except Exception as e:
            raise CreditCardException(e, sys)

    def create_endpoint(self):
        try:
            # Create online endpoint
            endpoint = ManagedOnlineEndpoint(
                name=AZURE_ML_ENDPOINT_NAME,
                description="Endpoint for CreditCard  predictions",
                auth_mode="key",
            )
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            return endpoint
        except Exception as e:
            raise CreditCardException(e, sys)

    def create_deployment(self, model_name: str, environment_name: str):
        try:
            # Create online deployment
            deployment = ManagedOnlineDeployment(
                name=AZURE_ML_DEPLOYMENT_NAME,
                endpoint_name=AZURE_ML_ENDPOINT_NAME,
                model=model_name,
                environment=environment_name,
                instance_type=AZURE_ML_INSTANCE_TYPE,
                instance_count=AZURE_ML_INSTANCE_COUNT,
            )
            self.ml_client.online_deployments.begin_create_or_update(
                deployment
            ).result()

            # Set deployment as default
            self.ml_client.online_endpoints.begin_update(
                name=AZURE_ML_ENDPOINT_NAME, traffic={AZURE_ML_DEPLOYMENT_NAME: 100}
            ).result()

            return deployment
        except Exception as e:
            raise CreditCardException(e, sys)

    def setup_azure_deployment(self, model_path: str):
        """Main function to set up the complete Azure ML deployment"""
        try:
            logging.info("Starting Azure ML setup")

            # Create environment
            logging.info("Creating Azure ML environment")
            env = self.create_environment()

            # Register model
            logging.info("Registering model")
            model = self.register_model(model_path)

            # Create endpoint
            logging.info("Creating endpoint")
            endpoint = self.create_endpoint()

            # Create deployment
            logging.info("Creating deployment")
            deployment = self.create_deployment(model.name, env.name)

            logging.info("Azure ML setup completed successfully")
            return {
                "endpoint_name": AZURE_ML_ENDPOINT_NAME,
                "deployment_name": AZURE_ML_DEPLOYMENT_NAME,
            }
        except Exception as e:
            logging.error(f"Azure ML setup failed: {str(e)}")
            raise CreditCardException(e, sys)
