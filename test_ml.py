import pytest, pickle, os
import pandas as pd

from ml.model import load_model
from train_model import model_path, compute_model_metrics, inference, cat_features, data_path
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_dataset():
    data_path = '/home/runner/work/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv'
    df = pd.read_csv(data_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    return X_train, y_train

# Implement the first test.
def test_data_shape():
    """
    Check data path and sizes
    """
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'census.csv')
    df = pd.read_csv(data_path)

    assert df.shape[0] > 0, 'Data set has no rows'
    assert df.shape[1] > 0, 'Data set has no columns'


# Implement the second test.
def test_model_algorithm():
    """
    Check that the model uses the expected algorithm.
    """
    model = load_model(model_path)
    assert isinstance(model, RandomForestClassifier), 'Model is not a Random Forest'


# Implement the third test.
def test_compute_model_metrics():
    """
    Check model metrics
    """
    X_train, y_train = train_dataset()

    model = pickle.load(open(model_path), 'rb')
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    assert round(precision, 4) == 0.8823
    assert round(recall, 4) == 0.6789
    assert round(fbeta, 4) == 0.7674

