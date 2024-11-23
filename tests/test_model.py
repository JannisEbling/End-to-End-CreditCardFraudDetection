import pytest
import pandas as pd
import numpy as np
from src.utils.ml_utils.model.estimator import CreditCardModel
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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


@pytest.fixture
def preprocessor():
    """Create a sample preprocessor"""
    return StandardScaler()


@pytest.fixture
def model():
    """Create a sample model"""
    return XGBClassifier(random_state=42)


def test_credit_card_model_initialization(preprocessor, model):
    """Test CreditCardModel initialization"""
    credit_card_model = CreditCardModel(preprocessor=preprocessor, model=model)
    assert credit_card_model is not None
    assert hasattr(credit_card_model, "model")
    assert hasattr(credit_card_model, "preprocessor")


def test_model_training(sample_data, preprocessor, model):
    """Test model training process"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    credit_card_model = CreditCardModel(preprocessor=preprocessor, model=model)

    # Fit preprocessor and transform data
    X_transformed = preprocessor.fit_transform(X)

    # Train model
    model.fit(X_transformed, y)

    assert hasattr(credit_card_model, "model")
    assert credit_card_model.model is not None


def test_model_prediction(sample_data, preprocessor, model):
    """Test model prediction"""
    X = sample_data.drop("target", axis=1)
    y = sample_data["target"]

    credit_card_model = CreditCardModel(preprocessor=preprocessor, model=model)

    # Fit preprocessor and transform data
    X_transformed = preprocessor.fit_transform(X)

    # Train model
    model.fit(X_transformed, y)

    # Make predictions
    predictions = model.predict(X_transformed)

    assert len(predictions) == len(X)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"feature1": [], "target": []}),
    ],
)
def test_model_invalid_input(invalid_data, preprocessor, model):
    """Test model behavior with invalid input"""
    credit_card_model = CreditCardModel(preprocessor=preprocessor, model=model)

    with pytest.raises(Exception):
        if invalid_data is None:
            credit_card_model.predict(None)
        else:
            credit_card_model.predict(invalid_data)
