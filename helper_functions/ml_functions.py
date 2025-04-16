import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List, Dict, Callable, Optional, Union, Tuple
from numpy.typing import NDArray


def adj_threshold_to_labels(model_probs: NDArray, threshold: float) -> NDArray:
    """
    Converts model probabilities into binary labels based on a threshold.
    """
    return (model_probs >= threshold).astype(int)


def reversed_feature(feature: NDArray) -> NDArray:
    """
    Applies reciprocal transformation to the input feature.
    """
    return np.power(feature.astype(float), -1)


def log_feature(feature: NDArray) -> NDArray:
    """
    Applies natural logarithm transformation to the input feature.
    """
    return np.log(feature.astype(float) + 1e-4)


def squared_feature(feature: NDArray) -> NDArray:
    """
    Squares the input feature.
    """
    return np.power(feature.astype(float), 2)


def cubic_feature(feature: NDArray) -> NDArray:
    """
    Cubes the input feature.
    """
    return np.power(feature.astype(float), 3)


class KmeansClustering(BaseEstimator, TransformerMixin):
    """
    Performs KMeans clustering on numerical features and appends the cluster label.
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()

    def fit(self, X: NDArray, y=None) -> "KmeansClustering":
        return self

    def transform(self, X: NDArray) -> NDArray:
        if self.kmeans:
            X_scaled = self.scaler.transform(X)
            clusters = self.kmeans.predict(X_scaled)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=50, random_state=42)
            clusters = self.kmeans.fit_predict(X_scaled)
        return np.c_[X, clusters]


class KmeansClusterDistance(BaseEstimator, TransformerMixin):
    """
    Appends distances to KMeans cluster centroids for each sample.
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.centroids: Optional[NDArray] = None
        self.col_labels: Optional[List[str]] = None
        self.scaler = StandardScaler()

    def fit(self, X: NDArray, y=None) -> "KmeansClusterDistance":
        return self

    def transform(self, X: NDArray) -> NDArray:
        if self.centroids is not None:
            X_scaled = self.scaler.transform(X)
            distances = self.kmeans.transform(X_scaled)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=50, random_state=42)
            distances = self.kmeans.fit_transform(X_scaled)
            self.centroids = distances
            self.col_labels = [f"Centroid_{i}" for i in range(distances.shape[1])]
        return np.c_[X, distances]


class NumericFeatureTransformation(BaseEstimator, TransformerMixin):
    """
    Applies a list of transformations to selected numerical features and retains
    only those transformations that increase correlation with the target.
    """

    def __init__(
        self,
        num_col_labels: List[str],
        num_col_idx: List[int],
        func_list: List[Callable[[NDArray], NDArray]],
        y: NDArray,
    ):
        self.num_col_labels = num_col_labels
        self.num_col_idx = num_col_idx
        self.col_labels: List[str] = []
        self.func_list = func_list
        self.y = y
        self.test_check = False

    def check_if_better(self, feature: NDArray, new_feature: NDArray) -> bool:
        """
        Checks if the new feature has stronger correlation with y than the original.
        """
        if new_feature.shape[0] == self.y.shape[0]:
            corr_orig = np.corrcoef(feature, self.y)[0, 1]
            corr_new = np.corrcoef(new_feature, self.y)[0, 1]
            return abs(round(corr_new, 3)) > abs(round(corr_orig, 3))
        else:
            self.test_check = True
            return False

    def fit(self, X: NDArray, y=None) -> "NumericFeatureTransformation":
        return self

    def transform(self, X: NDArray) -> NDArray:
        for label, idx in zip(self.num_col_labels, self.num_col_idx):
            original_feature = X[:, idx]
            for func in self.func_list:
                new_feature = func(original_feature)
                new_label = f"{label}_{func.__name__.split('_')[0]}"
                if (
                    self.check_if_better(original_feature, new_feature)
                    and new_label not in self.col_labels
                    and not self.test_check
                ):
                    self.col_labels.append(new_label)
                    X = np.c_[X, new_feature]
                elif self.test_check and new_label in self.col_labels:
                    X = np.c_[X, new_feature]
        return X


def data_preparation(user_input: Dict[str, Union[str, float]]) -> pd.DataFrame:
    """
    Converts a user input dictionary into a one-row DataFrame,
    converting numeric values where possible.
    """
    for key, value in user_input.items():
        try:
            user_input[key] = float(value)
        except (ValueError, TypeError):
            pass  # Keep the original value if conversion fails
    return pd.DataFrame([user_input])


def clf_model_prediction(
    model: BaseEstimator, pipe: Pipeline, data: pd.DataFrame
) -> NDArray:
    """
    Applies preprocessing and returns classification probability predictions.
    """
    data_tr = pipe.transform(data)
    return model.predict_proba(data_tr)[:, 1]


def reg_model_prediction(
    model: BaseEstimator, pipe: Pipeline, data: pd.DataFrame
) -> NDArray:
    """
    Applies preprocessing and returns regression predictions.
    """
    data_tr = pipe.transform(data)
    return model.predict(data_tr)


def load_model_and_pipe(
    PIPE_PATH: str, pipe_name: str, MODEL_PATH: str, model_name: str
) -> Tuple[Pipeline, BaseEstimator]:
    """
    Loads a preprocessing pipeline and ML model from disk.

    Args:
        pipe_name (str): Filename of the pipeline (e.g., "stroke_final_pipeline.pkl")
        model_name (str): Filename of the model (e.g., "stroke_catboost.pkl")

    Returns:
        tuple: (Pipeline, Model)
    """
    with open(os.path.join(PIPE_PATH, pipe_name), "rb") as pipe_file, open(
        os.path.join(MODEL_PATH, model_name), "rb"
    ) as model_file:
        pipe = joblib.load(pipe_file)
        model = joblib.load(model_file)
    return pipe, model
