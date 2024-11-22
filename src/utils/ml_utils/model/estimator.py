from src.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os
import sys
import pandas as pd
import numpy as np
from src.exception.exception import CreditCardException
from src.logging.logger import logging
from src.constant.training_pipeline import TARGET_COLUMN


class CreditCardModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            # Initialize azure_predictor only when needed
            self._azure_predictor = None
        except Exception as e:
            raise CreditCardException(e, sys)

    @property
    def azure_predictor(self):
        if self._azure_predictor is None:
            from src.cloud.azure_predictor import AzurePredictor
            self._azure_predictor = AzurePredictor()
        return self._azure_predictor

    def predict(self, x):
        try:
            # Get the feature names the preprocessor was trained on
            if hasattr(self.preprocessor, "feature_names_in_"):
                required_features = self.preprocessor.feature_names_in_
            else:
                # If using the first step of the pipeline
                required_features = self.preprocessor.steps[0][1].feature_names_in_

            # Convert input to DataFrame if it's not already
            if not isinstance(x, pd.DataFrame):
                if isinstance(x, np.ndarray):
                    x = pd.DataFrame(x, columns=required_features)
                else:
                    x = pd.DataFrame(x)

            # Remove the target column if it exists in the input
            if TARGET_COLUMN in x.columns:
                x = x.drop(columns=[TARGET_COLUMN])

            # Reorder columns to match training data
            missing_cols = set(required_features) - set(x.columns)
            extra_cols = set(x.columns) - set(required_features)

            if missing_cols:
                raise ValueError(f"Missing required features: {missing_cols}")

            if extra_cols:
                logging.warning(f"Extra features will be ignored: {extra_cols}")
                x = x[required_features]

            # Ensure column order matches training data
            x = x[required_features]

            transformed_features = self.preprocessor.transform(x)
            predictions = self.model.predict(transformed_features)
            return predictions

        except Exception as e:
            raise CreditCardException(e, sys)

    def __getstate__(self):
        """Custom serialization method"""
        state = self.__dict__.copy()
        # Don't pickle the azure_predictor
        state['_azure_predictor'] = None
        return state

    def __setstate__(self, state):
        """Custom deserialization method"""
        self.__dict__.update(state)
