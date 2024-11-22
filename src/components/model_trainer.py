import os
import sys

from src.exception.exception import CreditCardException
from src.logging.logger import logging

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig


from src.utils.ml_utils.model.estimator import CreditCardModel
from src.utils.main_utils.utils import save_object, load_object
from src.utils.main_utils.utils import (
    load_numpy_array_data,
    evaluate_models,
)
from src.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier
import mlflow
from urllib.parse import urlparse

# Configure MLflow
mlflow.set_tracking_uri("file:///mlruns")

os.environ["MLFLOW_TRACKING_URI"] = "file:///mlruns"
os.environ["MLFLOW_TRACKING_USERNAME"] = ""
os.environ["MLFLOW_TRACKING_PASSWORD"] = ""


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CreditCardException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        try:
            mlflow.set_registry_uri("file:///mlruns")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Convert the model to a simple sklearn model if it's a wrapped version
            if hasattr(best_model, "get_params"):
                model_params = best_model.get_params()
                # Remove any Azure ML specific parameters
                model_params = {
                    k: v
                    for k, v in model_params.items()
                    if not isinstance(v, type) and str(type(v)).find("azure.") == -1
                }
                best_model.set_params(**model_params)

            with mlflow.start_run():
                f1_score = classificationmetric.f1_score
                precision_score = classificationmetric.precision_score
                recall_score = classificationmetric.recall_score

                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision", precision_score)
                mlflow.log_metric("recall_score", recall_score)

                # Save only the model's state dict if possible
                if hasattr(best_model, "state_dict"):
                    mlflow.sklearn.log_model(best_model.state_dict(), "model")
                else:
                    mlflow.sklearn.log_model(best_model, "model")

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    if hasattr(best_model, "state_dict"):
                        mlflow.sklearn.log_model(
                            best_model.state_dict(),
                            "model",
                            registered_model_name=str(type(best_model).__name__),
                        )
                    else:
                        mlflow.sklearn.log_model(
                            best_model,
                            "model",
                            registered_model_name=str(type(best_model).__name__),
                        )
        except Exception as e:
            logging.error(f"Error in MLflow tracking: {str(e)}")
            # Continue execution even if MLflow tracking fails
            pass

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
            "XGBoost": XGBClassifier(),
        }
        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy"],
            },
            "Random Forest": {"n_estimators": [16, 64, 128]},
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01],
                "subsample": [0.7, 0.9],
                "n_estimators": [32, 64],
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [32, 64],
            },
            "XGBoost": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [32, 64],
                "max_depth": [3, 7],
                "subsample": [0.7, 0.9],
            },
        }
        model_report: dict = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(
            y_true=y_train, y_pred=y_train_pred
        )

        ## Track the experiements with mlflow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(
            y_true=y_test, y_pred=y_test_pred
        )

        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        try:
            # Create model directory if it doesn't exist
            os.makedirs("final_model", exist_ok=True)
            
            # Save the model and preprocessor separately
            save_object("final_model/model.pkl", best_model)
            save_object("final_model/preprocessor.pkl", preprocessor)
            
            # Save the complete pipeline
            CreditCard_Model = CreditCardModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=CreditCard_Model)
            
            logging.info("Model, preprocessor and pipeline saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise CreditCardException(e, sys)
        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise CreditCardException(e, sys)