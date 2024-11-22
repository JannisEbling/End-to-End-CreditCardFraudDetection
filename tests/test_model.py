import pytest
import pandas as pd
import numpy as np
from src.utils.ml_utils.model.estimator import NetworkModel
from src.components.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100

    data = {
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(0, 1, n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }
    return pd.DataFrame(data)


def test_network_model_initialization():
    """Test NetworkModel initialization"""
    model = NetworkModel()
    assert model is not None
    assert hasattr(model, "model")


def test_model_training(sample_data):
    """Test model training process"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    model = NetworkModel()
    model.train(X, y)

    assert hasattr(model, "model")
    assert model.model is not None


def test_model_prediction(sample_data):
    """Test model prediction"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    model = NetworkModel()
    model.train(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)


def test_model_trainer_initialization():
    """Test ModelTrainer initialization"""
    trainer = ModelTrainer()
    assert trainer is not None


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"feature1": [], "target": []}),
    ],
)
def test_model_invalid_input(invalid_data):
    """Test model behavior with invalid input"""
    model = NetworkModel()
    with pytest.raises(Exception):
        if invalid_data is None:
            model.train(None, None)
        else:
            X = (
                invalid_data.drop("target", axis=1)
                if not invalid_data.empty
                else invalid_data
            )
            y = invalid_data["target"] if "target" in invalid_data else pd.Series()
