import lightgbm as lgb
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR


class LGBMAlphaModel:
    def __init__(self, hyperparams=None):
        """
        Initialize the LightGBM model with given hyperparameters.
        :param hyperparams: dict, LightGBM hyperparameters
        """
        self.hyperparams = hyperparams or {}
        self.model = None

    def train(self, X, y, val_data=None, early_stopping_rounds=50, verbose_eval=100):
        """
        Train the LightGBM model.
        :param X: np.ndarray or pd.DataFrame, training features
        :param y: np.ndarray or pd.Series, training labels
        :param val_data: tuple (X_val, y_val), optional validation data
        :param early_stopping_rounds: int, early stopping rounds
        :param verbose_eval: int, print frequency
        """
        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        valid_names = ["train"]

        if val_data is not None:
            X_val, y_val = val_data
            val_set = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(val_set)
            valid_names.append("valid")

        self.model = lgb.train(
            self.hyperparams,
            train_set,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=(
                early_stopping_rounds if val_data is not None else None
            ),
            verbose_eval=verbose_eval,
        )

    def predict(self, X):
        """
        Predict using the trained LightGBM model.
        :param X: np.ndarray or pd.DataFrame, features to predict
        :return: np.ndarray, predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class RandomForestAlphaModel:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams or {}
        self.model = RandomForestRegressor(**self.hyperparams)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class GradientBoostingAlphaModel:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams or {}
        self.model = GradientBoostingRegressor(**self.hyperparams)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class AdaBoostAlphaModel:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams or {}
        self.model = AdaBoostRegressor(**self.hyperparams)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class RidgeAlphaModel:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams or {}
        self.model = Ridge(**self.hyperparams)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class LassoAlphaModel:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams or {}
        self.model = Lasso(**self.hyperparams)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class SVRAlphaModel:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams or {}
        self.model = SVR(**self.hyperparams)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
