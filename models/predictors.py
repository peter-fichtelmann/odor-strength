from typing import Callable
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import CoralLoss
from functools import partial
import copy
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data import build_dataloader
from chemprop.models.model import MPNN
from chemprop.nn.message_passing.base import BondMessagePassing, AtomMessagePassing
from chemprop.nn.agg import MeanAggregation, SumAggregation, NormAggregation, AttentiveAggregation
from chemprop.nn import BinaryClassificationFFN, RegressionFFN, MulticlassClassificationFFN
from chemprop import featurizers
from lightning import pytorch as pl
import os
import logging
from datetime import datetime
import uuid
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.ERROR)
from utility.metrics import Metrics, TorchMetrics
from utility.label_binarizer import LabelBinarizer


class Predictor(ABC):
    """Abstract base class for all predictors. Each class should implement the fit and predict methods."""
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray|None = None, y_val:np.ndarray|None = None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def reset(self):
        pass


class PredictorBase(Predictor):
    """Base class for all predictors that provides common functionality.

    Methods:
        calculate_sample_weight(y): Calculates sample weights based on the frequency of each label.
        get_standardizer(standardizer_name): Returns a sklearn standardizer based on the provided name.
    """
    def calculate_sample_weight(self, y: np.ndarray) -> np.ndarray:
        """Calculates sample weights based on the frequency of each label.
        
        Parameters:
            y (np.ndarray): The target variable array.
        
        Returns:
            np.ndarray: A 1D array of sample weights.
        
        Example:
            >>> y = np.array([0, 1, 1, 2, 2, 2])
            >>> sample_weights = calculate_sample_weight(y)
            >>> print(sample_weights)
            array([2.        , 1.        , 1.        , 0.66666667, 0.66666667, 0.66666667])
            """
        if len(y.shape) > 1:
            y = np.sum(y, axis=1)
        sample_weight = np.zeros_like(y, dtype=float)
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            sample_weight[y == u] = 1 / c
        sample_weight = sample_weight * len(y) / np.sum(sample_weight)
        return sample_weight
    
    def get_standardizer(self, standardizer_name: str)-> object:
        """Returns a sklearn standardizer based on the provided name.
        Parameters:
            standardizer_name (str): The name of the standardizer to use. Options are 'minmax', 'standard', 'robust', 'yeo-johnson'."""
        if standardizer_name == 'minmax':
            return MinMaxScaler()
        elif standardizer_name == 'standard':
            return StandardScaler()
        elif standardizer_name == 'robust':
            return RobustScaler()
        elif standardizer_name == 'yeo-johnson':
            return PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError(f"Unknown standardizer: {standardizer_name}. Available options: 'minmax', 'standard', 'robust', 'yeo-johnson'.")
        
    def reset(self):
        self.__init__(**self._init_kwargs)


class Average(PredictorBase):
    """Predictor that always predicts the average of the target variable."""
    def __init__(self):
        self.average = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs):
        self.average = y.mean()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.average] * len(X))
    
    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def reset(self):
        pass
    
    
class LogisticRegressionPredictor(PredictorBase):
    """Predictor that uses Logistic Regression from sklearn with multioutputclassifier for classification tasks.
    Parameters:
        sample_weight (bool): default=True
            Whether to use sample weights.
        binarize_labels (bool): default=False
            Whether to binarize labels corresponding to a ordinary regression task (shape classes -1 binary labels).
        dealing_with_incosistency (str): {'sum', 'max'}, default='sum'
            How to deal with label inconsistency ('sum', 'max') if the labels are binarized for ordinary regression.
        standardizer_name (str|None): {'minmax', 'standard', 'robust', 'yeo-johnson', None}, default='standard'
            The sklearn standardizer to use for feature scaling.
        kwargs: Additional keyword arguments for the LogisticRegression model.
    """
    def __init__(self, sample_weight: bool = True, binarize_labels: bool = False, dealing_with_incosistency: str = 'sum', standardizer_name: str|None = 'standard', **kwargs):
        self._init_kwargs = {'sample_weight': sample_weight, 'binarize_labels': binarize_labels, 'dealing_with_incosistency': dealing_with_incosistency, 'standardizer_name': standardizer_name, **kwargs}
        self.model = LogisticRegression(**kwargs)
        self.sample_weight = sample_weight
        self.binarize_labels = binarize_labels
        self.label_binarizer = LabelBinarizer()
        self.dealing_with_incosistency = dealing_with_incosistency
        if standardizer_name:
            self.standardizer = self.get_standardizer(standardizer_name)
        self.standardizer_name = standardizer_name

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs):
        if self.standardizer_name:
            X = self.standardizer.fit_transform(X)
        if self.sample_weight:
            sample_weight = self.calculate_sample_weight(y)
        else:
            sample_weight = None
        if self.binarize_labels:
            y = self.label_binarizer.binarize_labels(y)
        if len(y.shape) > 1 and not isinstance(self.model, MultiOutputClassifier):
            self.model = MultiOutputClassifier(self.model, n_jobs=-1)
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.standardizer_name:
            X = self.standardizer.transform(X)
        if self.binarize_labels:
            predictions = self.model.predict(X)
            predictions = self.label_binarizer.inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency)
        else:
            predictions = self.model.predict(X)
        return predictions
    
    def save(self, path: str):
        """Saves the model to the specified path."""
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Loads the model from the specified path."""
        import joblib
        self.model = joblib.load(path)


class GaussianProcessClassifierPredictor(PredictorBase):
    """Predictor that uses Gaussian Process Classifier from sklearn with various kernels.
    Parameters:
        binarize_labels (bool): default=False
            Whether to binarize labels corresponding to a ordinary regression task (shape classes -1 binary labels).
        dealing_with_incosistency (str): {'sum', 'max'}, default='sum'
            How to deal with label inconsistency ('sum', 'max') if the labels are binarized for ordinary regression.
        standardizer_name (str|None): {'minmax', 'standard', 'robust', 'yeo-johnson', None}, default='standard'
            The sklearn standardizer to use for feature scaling.
        kernel_name (str): {'RBF', 'DotProduct', 'Matern', 'RationalQuadratic'}, default='RBF'
            The kernel to use for the Gaussian Process Classifier.
        kernel_hyperparameters (dict): default={}
            Hyperparameters for the kernel.
        kwargs: Additional keyword arguments for the GaussianProcessClassifier model."""
        
    def __init__(self,
                 binarize_labels: bool=False,
                 dealing_with_incosistency: str = 'sum',
                 standardizer_name: str|None = 'standard',
                 kernel_name: str = 'RBF',
                 kernel_hyperparameters: dict = {},
                 **kwargs
                 ):
        self._init_kwargs = {
            'binarize_labels': binarize_labels,
            'dealing_with_incosistency': dealing_with_incosistency,
            'standardizer_name': standardizer_name,
            'kernel_name': kernel_name,
            'kernel_hyperparameters': kernel_hyperparameters,
            **kwargs
        }
        kwargs['kernel'] = self.get_kernel(kernel_name, kernel_hyperparameters)
        params = kwargs.copy()
        if 'kernel_hyperparameters' in params:
            params.pop('kernel_hyperparameters')
        self.model = GaussianProcessClassifier(**params)
        self.binarize_labels = binarize_labels
        self.label_binarizer = LabelBinarizer()
        self.dealing_with_incosistency = dealing_with_incosistency
        if standardizer_name:
            self.standardizer = self.get_standardizer(standardizer_name)
        self.standardizer_name = standardizer_name

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs):
        if self.standardizer_name:
            X = self.standardizer.fit_transform(X)
        if self.binarize_labels:
            y = self.label_binarizer.binarize_labels(y)
        self.model.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.standardizer_name:
            X = self.standardizer.transform(X)
        if self.binarize_labels:
            predictions = self.model.predict(X)
            predictions = self.label_binarizer.inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency)
        else:
            predictions = self.model.predict(X)
        return predictions
    
    def get_kernel(self, kernel_name, kernel_hyperparameters):
        if kernel_name == 'RBF': 
            return RBF(length_scale=kernel_hyperparameters.get('length_scale', 1.0))
        elif kernel_name == 'DotProduct':
            return DotProduct(sigma_0=kernel_hyperparameters.get('sigma_0', 1.0))
        elif kernel_name == 'Matern':
            return Matern(length_scale=kernel_hyperparameters.get('length_scale', 1.0), nu=kernel_hyperparameters.get('nu', 1.5))
        elif kernel_name == 'RationalQuadratic':
            return RationalQuadratic(length_scale=kernel_hyperparameters.get('length_scale', 1.0), alpha=kernel_hyperparameters.get('alpha', 1.0))
        else:
            None

    def save(self, path: str):
        """Saves the model to the specified path."""
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Loads the model from the specified path."""
        import joblib
        self.model = joblib.load(path)

      
class RandomForestClassifierPredictor(PredictorBase):
    """Predictor that uses Random Forest Classifier from sklearn with multioutputclassifier for classification tasks.
    Parameters:
        sample_weight (bool): default=True
            Whether to use sample weights.
        binarize_labels (bool): default=False
            Whether to binarize labels corresponding to a ordinary regression task (shape classes -1 binary labels).
        dealing_with_incosistency (str): {'sum', 'max'}, default='sum'
            How to deal with label inconsistency ('sum', 'max') if the labels are binarized for ordinary regression.
        kwargs: Additional keyword arguments for the RandomForestClassifier model."""
    def __init__(self, sample_weight: bool = True, binarize_labels=False, dealing_with_incosistency='sum', **kwargs):
        self._init_kwargs = {'sample_weight': sample_weight, 'binarize_labels': binarize_labels, 'dealing_with_incosistency': dealing_with_incosistency, **kwargs}
        self.model = RandomForestClassifier(**kwargs)
        self.sample_weight = sample_weight
        self.binarize_labels = binarize_labels
        self.label_binarizer = LabelBinarizer()
        self.dealing_with_incosistency = dealing_with_incosistency

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs):
        if self.sample_weight:
            sample_weight = self.calculate_sample_weight(y)
        else:
            sample_weight = None
        if self.binarize_labels:
            y = self.label_binarizer.binarize_labels(y)
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.binarize_labels:
            predictions = self.model.predict(X)
            predictions = self.label_binarizer.inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency)
        else:
            predictions = self.model.predict(X)
        return predictions
    
    def save(self, path: str):
        """Saves the model to the specified path."""
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Loads the model from the specified path."""
        import joblib
        self.model = joblib.load(path)
    

class RandomForestRegressorPredictor(PredictorBase):
    """Predictor that uses Random Forest Regressor from sklearn with multioutputclassifier.
    Parameters:
        sample_weight (bool): default=True
            Whether to use sample weights.
        kwargs: Additional keyword arguments for the RandomForestRegressor model."""
    def __init__(self, sample_weight: bool = True, **kwargs):
        self._init_kwargs = {'sample_weight': sample_weight, **kwargs}
        self.model = RandomForestRegressor(**kwargs)
        self.sample_weight = sample_weight

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs):
        if self.sample_weight:
            sample_weight = self.calculate_sample_weight(y)
        else:
            sample_weight = None
        self.model.fit(X, y, sample_weight=sample_weight, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def save(self, path: str):
        """Saves the model to the specified path."""
        import joblib
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        """Loads the model from the specified path."""
        import joblib
        self.model = joblib.load(path)
    

class RandomForestPredictor(PredictorBase):
    # model which is classifier or regressor based on a define objective
    def __init__(self, objective: str|None = 'regression', **kwargs):
        self._init_kwargs = {
            'objective': objective,
            **kwargs
        }
        self.model = RandomForestClassifierPredictor(**kwargs) if objective == 'classification' else RandomForestRegressorPredictor(**kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs):
        self.model.fit(X, y, X_val=X_val, y_val=y_val, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def save(self, path: str):
        self.model.save(path)
    
    def load(self, path: str):
        self.model.load(path)


class XGBoostPredictor(PredictorBase):
    """Predictor that uses XGBoost for regression or classification tasks.
    Parameters:
        sample_weight (bool): default=True
            Whether to use sample weights.
        binarize_labels (bool): default=False
            Whether to binarize labels corresponding to a ordinary regression task (shape classes -1 binary labels).
        dealing_with_incosistency (str): {'sum', 'max'}, default='sum'
            How to deal with label inconsistency ('sum', 'max') if the labels are binarized for ordinary regression.
        objective (str|None): default=None
            The objective function to use for the XGBoost model. If None, the default objective is used.
    Methods:
        fit(X, y, X_val=None, y_val=None, **kwargs): Fits the XGBoost model to the training data. Use validation data for early stopping.
        predict(X): Predicts the target variable for the given input data.
        get_loss_function(objective): Returns the appropriate loss function based on the objective string.
        get_callbacks(early_stopping_rounds, custom_metric_name): Returns a list of callbacks for early stopping and custom metrics.
        get_custom_metric(metric_name): Returns the appropriate custom metric function based on the metric name and if the metric should be maximized or minimized.
            """
    def __init__(self,
                 sample_weight: bool = True,
                 binarize_labels: bool = False,
                 dealing_with_incosistency: str = 'sum',
                 objective: str|None = None,
                 **kwargs
                 ):
        self._init_kwargs = {
            'sample_weight': sample_weight,
            'binarize_labels': binarize_labels,
            'dealing_with_incosistency': dealing_with_incosistency,
            'objective': objective,
            **kwargs
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics = Metrics()
        self.label_binarizer = LabelBinarizer()
        self.dealing_with_incosistency = dealing_with_incosistency
        self.objective = objective
        if self.objective:
            kwargs['objective'] = self.get_loss_function(self.objective)
        kwargs['disable_default_eval_metric'] = True
        self.kwargs = kwargs
        self.sample_weight = sample_weight
        self.binarize_labels = binarize_labels
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None = None, y_val:np.ndarray|None = None, **kwargs) -> int:
        params = self.kwargs.copy()
        additional_model_kwargs = kwargs
        if self.binarize_labels:
            y = self.label_binarizer.binarize_labels(y)
        if self.sample_weight:
            weights = self.calculate_sample_weight(y)
        else:
            weights = None
        dtrain = xgb.DMatrix(X, label=y, weight=weights)
        eval_set = [(dtrain, 'train')]
        if 'custom_metric' in self.kwargs:
            custom_metric_name = params.pop('custom_metric')
            additional_model_kwargs['custom_metric'] = self.get_custom_metric(custom_metric_name)
        else:
            custom_metric_name = 'mse_macro'
            additional_model_kwargs['custom_metric'] = self.mse_macro
        if isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray):
            if self.binarize_labels:
                y_val = self.label_binarizer.binarize_labels(y_val)
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_set.append((dval, 'validation'))
            if 'early_stopping_rounds' in self.kwargs:
                callbacks = self.get_callbacks(params.pop('early_stopping_rounds'), custom_metric_name)
                additional_model_kwargs['callbacks'] = callbacks
        if 'num_boost_round' in self.kwargs:
            additional_model_kwargs['num_boost_round'] = params.pop('num_boost_round')
        if 'verbose_eval' in self.kwargs:
            additional_model_kwargs['verbose_eval'] = params.pop('verbose_eval')
        if not isinstance(self.kwargs.get('objective', None), str):
            additional_model_kwargs['obj'] = params.pop('objective')
        if 'booster' in params:
            if params['booster'] == 'gbtree' and self.binarize_labels:
                if params.get('multi_strategy', None) == 'multi_output_tree':
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        params['device'] = self.device
        self.model = xgb.train(
            params,
            dtrain,
            evals=eval_set,
            **additional_model_kwargs
        )
        if isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray):
            best_boost_round = self.model.best_iteration
            return best_boost_round
        # if hasattr(self.model, 'best_iteration'): # slicing not supported for gblinear
        #     if additional_model_kwargs['num_boost_round'] > self.model.best_iteration + 1:
        #         # print(slice(0, self.model.best_iteration + 1, 1))
        #         self.model = self.model[0:self.model.best_iteration + 1] # alternative iteration range in prediction
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.binarize_labels:
            predictions = self.model.predict(xgb.DMatrix(X), iteration_range=(0, self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') else 0))
            if predictions.shape[0] != X.shape[0]:
                predictions = predictions.reshape(predictions.shape[0], predictions.shape[0] // X.shape[0])
            if self.objective == 'weighted_logloss':
                predictions = 1.0 / (1.0 + np.exp(-predictions))
            predictions = self.label_binarizer.inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency)
        else:
            predictions = self.model.predict(xgb.DMatrix(X), iteration_range=(0, self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') else 0))
            if self.objective == 'weighted_logloss':
                predictions = 1.0 / (1.0 + np.exp(-predictions))
        return predictions
    
    def get_loss_function(self, objective: str) -> object|str:
        if objective == 'weighted_mse':
            objective = self.weighted_mse_obj
        elif objective == 'weighted_logloss':
            objective = self.weighted_logloss
        return objective
    
    def get_callbacks(self, early_stopping_rounds: int|None, custom_metric_name: str|None) -> list:
        callbacks = []
        if early_stopping_rounds:
            if custom_metric_name == 'f1_score':
                maximize = True
            else:
                maximize = False
            early_stopping = xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,  # Number of rounds without improvement before stopping
                metric_name=custom_metric_name, 
                data_name='validation',
                maximize=maximize,
                save_best=False, 
            )
            callbacks.append(early_stopping)
        return callbacks
    
    def get_custom_metric(self, metric_name: str) -> object|str:
        if metric_name == 'mse_macro':
            return self.mse_macro
        elif metric_name == 'f1_score':
            return self.f1_score
        else:
            return metric_name

    def weighted_mse_obj(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        weights = dtrain.get_weight()
        if weights is None:
            weights = np.ones_like(preds)
        labels = dtrain.get_label()
        labels = np.reshape(labels, preds.shape)
        # if self.sample_weight:
        #     weights = self.calculate_sample_weight(labels)
        # else:
        #     weights = np.ones_like(labels)
        grad = weights * (preds - labels)
        # grad = weights * (preds - labels)
        hess = weights
        # try SLE
        # preds[preds < -1] = -1 + 1e-6
        # grad = (np.log1p(preds) - np.log1p(labels)) / (preds + 1)
        # hess = ((-np.log1p(preds) + np.log1p(labels) + 1) /
        #     np.power(preds + 1, 2))
        return grad, hess

    def weighted_logloss(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        weights = dtrain.get_weight()
        if weights is None:
            weights = np.ones_like(preds)
        if len(labels.shape) > 1:
            weights = np.reshape(weights, (-1, 1))
        labels = dtrain.get_label()
        labels = np.reshape(labels, preds.shape)
        # if self.sample_weight:
        #     weights = self.calculate_sample_weight(labels)
        #     if len(labels.shape) > 1:
        #         weights = np.reshape(weights, (-1, 1))
        # else:
        #     weights = np.ones_like(labels)
        # weights = np.zeros_like(labels)
        # for unique_label in np.unique(labels):
        #     if len(labels.shape) == 1:
        #         weights[labels == unique_label] = 1 / np.sum(labels == unique_label)
        #     else:
        #         for i in range(labels.shape[1]):
        #             weights[labels == unique_label, i] = 1 / np.sum(labels[labels == unique_label, i])
        preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid
        eps = 1e-10  # to avoid division by zero
        # y=1 > grad = -1/x
        squared_preds = np.square(preds)
        grad = weights * (preds - labels) / (preds - squared_preds + eps)
        hess = weights * (squared_preds - 2 * preds * labels + labels) / ((np.square(1 - squared_preds) * squared_preds) + eps)
        # with sigmoid included:
        # exp_preds = np.exp(preds)
        # grad = - weights * (exp_preds * (labels - 1) + labels) / (exp_preds + 1 + eps)
        # hess = - weights * exp_preds / (np.square(exp_preds + 1) + eps)
        # grad = weights * (preds - labels)
        # hess = weights * preds * (1.0 - preds)
        # grad = weights * (preds - labels)
        # hess = weights
        return grad, hess
    
    def mse_macro(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        labels = dtrain.get_label()
        num_classes = labels.shape[0] // preds.shape[0]
        if num_classes > 1:
            labels = labels.reshape(preds.shape[0], num_classes)
        if self.objective == 'weighted_logloss':
            preds = 1.0 / (1.0 + np.exp(-preds))
        preds= np.round(preds).astype(int)
        if len(np.squeeze(preds).shape) > 1:
            preds = self.label_binarizer.inverse_binarize_labels(preds, dealing_with_incosistency=self.dealing_with_incosistency)
        if len(np.squeeze(labels).shape) > 1:
            labels = self.label_binarizer.inverse_binarize_labels(labels, dealing_with_incosistency=self.dealing_with_incosistency)
        mean_mse_macro = self.metrics.calculate_mse_macro(labels, preds)[0]
        return 'mse_macro', float(mean_mse_macro)
    
    def f1_score(self, preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        labels = dtrain.get_label()
        num_classes = labels.shape[0] // preds.shape[0]
        if num_classes > 1:
            labels = labels.reshape(preds.shape[0], num_classes)
        if self.objective == 'weighted_logloss':
            preds = 1.0 / (1.0 + np.exp(-preds))
        preds = np.round(preds).astype(int)
        f1 = self.metrics.calculate_f1_score(labels, preds)[0]
        return 'f1_score', float(f1)
    
    def save(self, path: str):
        """Saves the model to the specified path."""
        self.model.save_model(path)

    def load(self, path: str):
        """Loads the model from the specified path."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        if hasattr(self.model, 'best_iteration'):
            self.model.best_iteration = self.model.best_ntree_limit


class MLP_Model(nn.Module):
    def __init__(self,
                 n_layers: int,
                 dim: int,
                 input_dim: int,
                 output_dim: int,
                 activation: torch.nn.modules.activation,
                 dropout: float,
                 ):
        super(MLP_Model, self).__init__()
        model_layers = []
        model_layers.append(nn.Linear(input_dim, dim))
        model_layers.append(activation)
        model_layers.append(nn.Dropout(dropout))
        for i in range(n_layers - 1):
            model_layers.append(nn.Linear(dim, dim))
            model_layers.append(activation)
            model_layers.append(nn.Dropout(dropout))
        model_layers.append(nn.Linear(dim, output_dim))
        self.model = nn.Sequential(*model_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class CoralModel(nn.Module):
    def __init__(self,
                 n_layers: int,
                 dim: int,
                 input_dim: int,
                 output_dim: int,
                 activation: torch.nn.modules.activation,
                 dropout: float,
                 ):
        super(CoralModel, self).__init__()
        model_layers = []
        model_layers.append(nn.Linear(input_dim, dim))
        model_layers.append(activation)
        model_layers.append(nn.Dropout(dropout))
        for i in range(n_layers - 1):
            model_layers.append(nn.Linear(dim, dim))
            model_layers.append(activation)
            model_layers.append(nn.Dropout(dropout))
        model_layers.append(CoralLayer(dim, output_dim + 1))
        self.model = nn.Sequential(*model_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    
class MLP_PredictorBase(PredictorBase):
    """Base class for MLP-based predictors.
    
    Parameters:
        sample_weight (bool): default=True
            Whether to use sample weights.
        binarize_labels (bool): default=False
            Whether to binarize labels corresponding to a ordinary regression task (shape classes -1 binary labels).
        dealing_with_incosistency (str): {'sum', 'max'}, default='sum'
            How to deal with label inconsistency ('sum', 'max') if the labels are binarized for ordinary regression.
        n_layers (int): default=5
            Number of layers in the MLP.
        dim (int): default=128
            Dimension of the hidden layers.
        batch_size (int): default=32
            Batch size for training.
        n_epochs (int): default=10
            Maximum number of epochs for training. The epochs can be reduced by early stopping.
        objective_name (str): {'mse', 'mae', 'cross_entropy', 'bce'}, default='mse'
            The objective function to use for the MLP model.
        learning_rate (float): default=0.001
            Learning rate for the optimizer.
        optimizer_name (str): {'adam', 'sgd', 'adamw', 'adagrad', 'rmsprop'}, default='adam'
            The optimizer to use for training the MLP model.
        optimizer_params (dict): default={}
            Additional parameters for the optimizer.
        training_scheduler_name (str|None): {None, 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'CyclicLR'} default=None
            The learning rate scheduler to use for training the MLP model. If None, no scheduler except a linear warmup scheduler is used.
        training_scheduler_params (dict): default={}
            Additional parameters for the learning rate scheduler.
        warmup_epochs (int): default=0
            Number of warmup epochs for the linear learning rate scheduler. If 0, no warmup is applied.
        activation (str): {'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'} default='relu'
            Activation function to use in the MLP layers.
        dropout (float): default=0.1
            Dropout rate to apply after each layer in the MLP.
        early_stopping_params (dict): default={}
            Parameters for early stopping:
                'early_stopping_rounds' (int): Number of rounds without improvement before stopping.
                'delta' (float): Minimum change in the monitored quantity to qualify as an improvement.
        weight_average (str|None): {None, 'swa', 'ema'} default=None
            Whether to use weight averaging during training. If 'swa', Stochastic Weight Averaging is used. If 'ema', Exponential Moving Average is used.
        metric_name (str): {'mse_macro', 'f1_score'}, default='mse_macro'
            The metric to use for evaluation during training.
        verbose_eval (bool): default=True
            Whether to print evaluation metrics during training.
        standardizer_name (str|None): {'minmax', 'standard', 'robust', 'yeo-johnson', None} default='standard'
            The sklearn standardizer to use for feature scaling. If None, no standardization is applied.
        
    Methods:
        fit(X, y, X_val=None, y_val=None, **kwargs): Fits the MLP model to the training data. Use validation data for early stopping.
        predict(X): Predicts the target variable for the given input data.
        train_cycle(
            model, epoch, device, dataloader, optimizer, warmup_scheduler, loss_function, weight_average
        ): Runs a training cycle for one epoch.
        validation_cycle(
            X, y, X_val, y_val, metric, metric_direction, sample_weights, epoch,
            early_stopping_rounds, delta, early_stopping_check, best_model, best_val_metric_value, best_epoch
        ): Runs a validation cycle for one epoch.
        get_loss_function(objective_name): Returns the appropriate loss function based on the objective name.
        get_activation(activation_name): Returns the appropriate activation function based on the activation name.
        get_optimizer(optimizer_name, model, optimizer_params): Returns the appropriate optimizer based on the optimizer name and model.
        get_training_scheduler(training_scheduler_name, optimizer, training_scheduler_params): Returns the appropriate training scheduler based on the training scheduler name and optimizer.
        get_metric(metric_name): Returns the appropriate metric function based on the metric name and if the metric should be maximized or minimized.
        get_standardizer(standardizer_name): Returns a sklearn standardizer based on the provided name
        get_callbacks(early_stopping_rounds, custom_metric_name): Returns a list of callbacks for early stopping and custom metrics.
        calculate_sample_weight(y): Calculates sample weights based on the frequency of each label.
        warmup_lr(epoch): Linear warmup learning rate scheduler function.
        transform_predictions(predictions): Transforms logits to probabilities for binary classification tasks.
    """
    def __init__(self,
                 sample_weight: bool = True,
                 binarize_labels: bool = False,
                 dealing_with_incosistency: str = 'sum',
                 n_layers: int = 5,
                 dim: int = 128,
                 batch_size: int = 32,
                 n_epochs: int = 10,
                 objective_name: str = 'mse',
                 learning_rate: float = 0.001,
                 optimizer_name: str = 'adam',
                 optimizer_params: dict = {},
                 training_scheduler_name: str|None = None,
                 training_scheduler_params: dict = {},
                 warmup_epochs: int = 0,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 early_stopping_params: dict = {},
                 weight_average: str|None = None,
                 metric_name: str = 'mse_macro',
                 verbose_eval: bool = True,
                 standardizer_name: str|None = 'standard',
                 input_dim: int = 256,
                 output_dim: int = 4,
                 ):
        self._init_kwargs = {
            'sample_weight': sample_weight,
            'binarize_labels': binarize_labels,
            'dealing_with_incosistency': dealing_with_incosistency,
            'n_layers': n_layers,
            'dim': dim,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'objective_name': objective_name,
            'learning_rate': learning_rate,
            'optimizer_name': optimizer_name,
            'optimizer_params': optimizer_params,
            'training_scheduler_name': training_scheduler_name,
            'training_scheduler_params': training_scheduler_params,
            'warmup_epochs': warmup_epochs,
            'activation': activation,
            'dropout': dropout,
            'early_stopping_params': early_stopping_params,
            'weight_average': weight_average,
            'metric_name': metric_name,
            'verbose_eval': verbose_eval,
            'standardizer_name': standardizer_name,
            'input_dim': input_dim,
            'output_dim': output_dim
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = 'cpu'
        print(f"Using device: {self.device}")
        self.model_architecture = MLP_Model    
        self.sample_weight = sample_weight
        self.label_binarizer = LabelBinarizer()
        self.binarize_labels = binarize_labels
        self.dealing_with_incosistency = dealing_with_incosistency
        self.n_layers = n_layers
        self.dim = dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate # 1e-3
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.training_scheduler_name = training_scheduler_name
        self.training_scheduler_params = training_scheduler_params
        self.objective_name = objective_name
        self.loss_function = self.get_loss_function(objective_name)
        self.warmup_epochs = warmup_epochs
        self.activation = self.get_activation(activation)
        self.dropout = dropout
        self.weight_average = weight_average
        self.early_stopping_params = early_stopping_params
        self.metric_name = metric_name
        self.metric, self.metric_direction = self.get_metric(metric_name)
        self.verbose_eval = verbose_eval
        if standardizer_name:
            self.standardizer = self.get_standardizer(standardizer_name)
        self.standardizer_name = standardizer_name
        self.dtype = torch.float32
        self.model = self.model_architecture(
            n_layers=self.n_layers,
            dim=self.dim,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=self.activation,
            dropout=self.dropout,
            ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs) -> int|None:
        def warmup_lr(epoch):  # Number of warm-up epochs
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs  # Gradually increase LR
            return 1.0  # Keep LR constant after warm-up
        sample_weights = self.calculate_sample_weight(y)
        sample_weights = sample_weights.reshape(-1, 1)
        sample_weights = torch.tensor(sample_weights, dtype=self.dtype).to(self.device)
        if self.standardizer_name:
            X = self.standardizer.fit_transform(X)
        if X_val is not None and y_val is not None:
            if self.binarize_labels:
                y_val = self.label_binarizer.binarize_labels(y_val)
            else:
                y_val = y_val.reshape(-1, 1)
            if self.standardizer_name:
                X_val = self.standardizer.transform(X_val)
            X_val = torch.tensor(X_val, dtype=self.dtype).to(self.device)
            y_val = torch.tensor(y_val, dtype=self.dtype).to(self.device)
        X = torch.tensor(X, dtype=self.dtype).to(self.device)
        if self.binarize_labels:
            y = self.label_binarizer.binarize_labels(y)
        else:
            y = y.reshape(-1, 1)
        y = torch.tensor(y, dtype=self.dtype).to(self.device)
        dataset = TensorDataset(X, y, sample_weights)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        input_dim = X.shape[1] if len(X.shape) > 1 else 1
        output_dim = y.shape[1] if len(y.shape) > 1 else 1
        self.model = self.model_architecture(
            n_layers=self.n_layers,
            dim=self.dim,
            input_dim=input_dim,
            output_dim=output_dim,
            activation=self.activation,
            dropout=self.dropout,
            ).to(self.device)
        self.optimizer = self.get_optimizer(self.optimizer_name, self.model, self.optimizer_params)
        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lr)
        training_scheduler = self.get_training_scheduler(self.training_scheduler_name, self.optimizer, self.training_scheduler_params)
# ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR
        if self.weight_average == 'swa':
            self.averaged_model = AveragedModel(self.model, device=self.device)
            self.swa_start = int(self.n_epochs * 0.5) # after 75% of training time including early stopping
            self.swa_scheduler = SWALR(self.optimizer, 0.5 * self.learning_rate, anneal_strategy="linear", anneal_epochs=5)
        elif self.weight_average == 'ema':
            self.averaged_model = torch.optim.swa_utils.AveragedModel(self.model, \
                            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
            self.swa_start = 0
        if self.early_stopping_params.get('early_stopping_rounds', None) is not None:
            early_stopping_rounds = self.early_stopping_params['early_stopping_rounds']
        else:
            early_stopping_rounds = 1000
        if 'delta' in self.early_stopping_params:
            delta = self.early_stopping_params['delta']
        else:
            delta = 0
        if self.metric_direction == 'maximize':
            best_val_metric = float('-inf')
        else:
            best_val_metric = float('inf')
        early_stopping_check = 0
        best_model = copy.deepcopy(self.model)
        best_epoch = None
        for epoch in range(self.n_epochs):
            self.model, warmup_scheduler = self.train_cycle(self.model, epoch, self.device, dataloader, self.optimizer, warmup_scheduler, self.loss_function, self.weight_average)
            # Need to fix the case of X_val and y_val being None
            if epoch == self.n_epochs - 1:
                early_stopping_check = early_stopping_rounds
            best_model, best_epoch, early_stopping_flag, early_stopping_check, best_val_metric, val_metric_value = self.validation_cycle(X, y, X_val, y_val, self.metric, self.metric_direction, sample_weights, epoch, early_stopping_rounds, delta, early_stopping_check, best_model, best_val_metric, best_epoch)
            if early_stopping_flag or epoch == self.n_epochs - 1:
                # def compare_model_weights(model1, model2):
                #     for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
                #         if not torch.equal(param1, param2):
                #             print(f"🔄 Weights changed in layer: {name1}")
                #             return False  # Weights are different
                #     print("✅ No changes in weights")
                #     return True  # Weights are identical
                # if self.weight_average:
                #     # print(compare_model_weights(self.model, best_model))
                #     for param, averaged_model_param in zip(self.model.parameters(), best_model.module.parameters()):
                #         param.data.copy_(averaged_model_param.data)
                #     # print(compare_model_weights(self.model, best_model))
                # else:
                # print(compare_model_weights(self.model, best_model))
                self.model = best_model
                # print(compare_model_weights(self.model, best_model))
                if self.verbose_eval:
                    print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, Best {self.metric_name}: {best_val_metric:.4f}")
                break
            if self.training_scheduler_name:
                if self.training_scheduler_name == 'ReduceLROnPlateau':
                    training_scheduler.step(val_metric_value)
                else:
                    training_scheduler.step()
        if self.weight_average:
            torch.optim.swa_utils.update_bn(dataloader, self.averaged_model) # or use_buffers=True in AveragedModel
            self.model = self.averaged_model
        return best_epoch

    def train_cycle(
        self,
        model: nn.Module,
        epoch: int,
        device: str,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        warmup_scheduler: LambdaLR,
        loss_function: Callable,
        weight_average: str | None
    ) -> tuple[nn.Module, LambdaLR]:
        model.train()
        for batch, batch_data in enumerate(dataloader):
            x = batch_data[0].to(device)
            y_batch = batch_data[1].to(device)
            sample_weight = batch_data[2].to(device)
            prediction = model(x)
            loss = loss_function(prediction, y_batch)
            loss = loss * sample_weight
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if weight_average:
            if epoch >= self.swa_start:
                self.averaged_model.update_parameters(model)
                if weight_average == 'swa':
                    self.swa_scheduler.step()
        warmup_scheduler.step()
        return model, warmup_scheduler

    def validation_cycle(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        metric: Callable,
        metric_direction: str,
        sample_weights: torch.Tensor,
        epoch: int,
        early_stopping_rounds: int,
        delta: float,
        early_stopping_check: int,
        best_model: nn.Module,
        best_val_metric_value: float,
        best_epoch: int | None
    ) -> tuple[nn.Module, int | None, bool, int, float, float]:
        # early stopping
        # delta: float, patience: int,
        self.model.eval()
        early_stopping_flag = False
        with torch.no_grad():
            # if self.weight_average and epoch >= self.swa_start:
            #     val_model = self.averaged_model
            # else:
            if X_val is not None and y_val is not None: # could add printing train loss for both None
                val_model = self.model
                prediction = val_model(X_val)
                raw_prediction = prediction.clone()
                prediction = self.transform_predictions(prediction)
                if self.binarize_labels:
                    prediction = self.label_binarizer.torch_inverse_binarize_labels(
                        prediction, dealing_with_incosistency=self.dealing_with_incosistency
                    )
                val_metric_value = metric(y_val, prediction, dealing_with_incosistency=self.dealing_with_incosistency)[0]
                if self.verbose_eval:
                    train_prediction = val_model(X)
                    train_loss = self.loss_function(train_prediction, y)
                    train_loss = train_loss * sample_weights
                    train_loss = torch.mean(train_loss).item()
                    val_loss = self.loss_function(raw_prediction, y_val)
                    val_loss = torch.mean(val_loss).item()
                    print(f"Epoch {epoch}: Train {train_loss:3f} Val {val_loss:3f} {metric.__name__} {val_metric_value:3f}")
                if metric_direction == 'minimize':
                    better_model_condition = (best_val_metric_value - val_metric_value) > delta
                elif self.metric_direction == 'maximize':
                    better_model_condition = (val_metric_value - best_val_metric_value) > delta
                else:
                    raise ValueError(f"Unknown metric direction: {metric_direction}. Available directions: minimize, maximize.")
                if better_model_condition:
                    best_model = copy.deepcopy(val_model)
                    best_val_metric_value = val_metric_value
                    early_stopping_check = 0
                    best_epoch = epoch
                elif epoch > self.warmup_epochs:
                    early_stopping_check += 1
                    if early_stopping_check >= early_stopping_rounds:
                        early_stopping_flag = True
                return best_model, best_epoch, early_stopping_flag, early_stopping_check, best_val_metric_value, val_metric_value
            else:
                train_prediction = self.model(X)
                if self.binarize_labels:
                    train_prediction = self.label_binarizer.torch_inverse_binarize_labels(train_prediction, dealing_with_incosistency=self.dealing_with_incosistency)
                train_metric_value = metric(y, train_prediction, dealing_with_incosistency=self.dealing_with_incosistency)[0]
            return self.model, None, False, 0, None, train_metric_value

    def get_optimizer(self, optimizer_name: str, model: nn.Module, optimizer_params: dict) -> torch.optim.Optimizer:
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=optimizer_params.get('weight_decay', 0.0), betas=(optimizer_params.get('betas', (0.9, 0.999))))
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=optimizer_params.get('weight_decay', 0.1), betas=(optimizer_params.get('betas', (0.9, 0.999))))
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=optimizer_params.get('weight_decay', 0.0), momentum=optimizer_params.get('momentum', 0.9), nesterov=optimizer_params.get('nesterov', False), dampening=optimizer_params.get('dampening', 0.0))
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.learning_rate, weight_decay=optimizer_params.get('weight_decay', 0.0), momentum=optimizer_params.get('momentum', 0.0), alpha=optimizer_params.get('alpha', 0.99))
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate, weight_decay=optimizer_params.get('weight_decay', 0.0), lr_decay=optimizer_params.get('lr_decay', 0.0))
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}. Available optimizers: adam, adamw, sgd, rmsprop, adagrad.")
        return optimizer
    
    def get_training_scheduler(self, training_scheduler_name: str|None, optimizer: torch.optim.Optimizer, training_scheduler_params: dict) -> torch.optim.lr_scheduler._LRScheduler | None:
        if training_scheduler_name is None:
            return None
        elif training_scheduler_name == 'ReduceLROnPlateau':
            if self.metric_direction == 'minimize':
                mode = 'min'
            else:
                mode = 'max'
            training_scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=training_scheduler_params.get('factor', 0.1), patience=self.early_stopping_params.get('early_stopping_rounds', 20)/2)
        elif training_scheduler_name == 'CosineAnnealingLR':
            training_scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        elif training_scheduler_name == 'CosineAnnealingWarmRestarts':                
            training_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=training_scheduler_params.get('T_0', 10), T_mult=training_scheduler_params.get('T_mult', 1))
        # elif training_scheduler_name == 'CyclicLR':
        #     training_scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=self.learning_rate, step_size_up=self.n_epochs // 2, mode='triangular2')
        else:
            raise ValueError(f"Unknown training scheduler name: {training_scheduler_name}. Available schedulers: ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR.")
        return training_scheduler

    def get_metric(self, metric_name: str) -> tuple[Callable, str]:
        if metric_name == 'mse_macro':
            metric = TorchMetrics().calculate_mse_macro
            metric_direction = 'minimize'
        elif metric_name == 'f1_score':
            metric = TorchMetrics().calculate_f1_score
            metric_direction = 'maximize'
        else:
            raise ValueError(f"Unknown metric name: {metric_name}. Available metrics: mse_macro, f1_score.")
        return metric, metric_direction

    def get_loss_function(self, objective_name: str) -> Callable:
        if objective_name == 'mse':
            loss_function = nn.MSELoss(reduction='none')
        elif objective_name == 'mae':
            loss_function = nn.L1Loss(reduction='none')
        # elif objective_name == 'cosine':
        #     def cosine_loss(x, y):
        #         return (1 - nn.functional.cosine_similarity(x, y, dim=-1)).mean()
        #         # return nn.CosineEmbeddingLoss()(x, y, torch.ones(x.shape[0]))
        #     # cosine_loss = nn.CosineEmbeddingLoss(margin=0.0)
        #     # loss_function = partial(cosine_loss, target=torch.tensor(1.0))
        #     loss_function = partial(cosine_loss)
        elif objective_name == 'crossentropy':
            loss_function = nn.CrossEntropyLoss(reduction='none')
        elif objective_name == 'binary_crossentropy':
            loss_function = nn.BCEWithLogitsLoss(reduction='none')
        elif objective_name == 'coral':
            loss_function = CoralLoss(reduction=None)
        return loss_function
    
    def get_activation(self, activation: str) -> torch.nn.modules.activation:
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky-relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}. Available functions: relu, leaky_relu, sigmoid, tanh, elu.")
        
    def transform_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.objective_name in ['binary_crossentropy', 'crossentropy', 'coral']:
            predictions = torch.sigmoid(predictions)
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
        self.model.eval()
        if self.standardizer_name:
            X = self.standardizer.transform(X)
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=self.dtype).to(self.device)
            predictions = self.model(X_tensor)
            predictions = self.transform_predictions(predictions)
            if self.binarize_labels:
                predictions = self.label_binarizer.torch_inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency)
            predictions = predictions.cpu().numpy()
            if len(predictions.shape) > 1:
                predictions = np.squeeze(predictions)
            # if self.binarize_labels:
            #     predictions = self.label_binarizer.inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency)
            return predictions
        
    def save(self, path: str):
        import joblib
        """Saves the model to the specified path."""
        joblib.dump(self.standardizer, path.split('.')[0] + '_standardizer.save')
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        import joblib
        self.standardizer = joblib.load(path.split('.')[0] + '_standardizer.save')
        self.model.load_state_dict(torch.load(path, weights_only=True))
        # self.model.load_state_dict(torch.load(path))
        self.model.eval()


class MLP_Predictor(MLP_PredictorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = MLP_Model


class CoralPredictor(MLP_PredictorBase):
    def __init__(self, **kwargs):
        kwargs['objective_name'] = 'coral'
        kwargs['binarize_labels'] = True
        super().__init__(**kwargs)
        self.model_architecture = CoralModel


class ChemPropMPNNPredictor(PredictorBase):
    """ChemProp MPNN Predictor for molecular property prediction.

    Parameters:
        binarize_labels (bool): default=False
            Whether to binarize labels corresponding to a ordinary regression task (shape classes -1 binary labels).
        dealing_with_incosistency (str): {'sum', 'max'}, default='sum'
            How to deal with label inconsistency ('sum', 'max') if the labels are binarized for ordinary regression.
        epochs (int): default=20
            Number of epochs for training the MPNN model.
        batch_size (int): default=64
            Batch size for training the MPNN model.
        message_passing (str): {'bond', 'atom', 'chemeleon'}, default='bond'
            Type of message passing to use in the MPNN model.
            If 'chemeleon' is selected, the pre-trained Chemeleon MPNN model (bond message passing) will be used and the following parameters will be ignored:
                mp_dim, bias, n_message_passing_iterations, messages_undirected_edges, mp_dropout, mp_activation.
        mp_dim (int): default=300
            Dimension of the message passing layer in the MPNN model.
        bias (bool): default=False
            Whether to use bias in the message passing layer.
        n_message_passing_iterations (int): default=3
            Number of message passing iterations in the MPNN model.
        messages_undirected_edges (bool): default=False
            Whether to use undirected edges in the message passing layer.
        mp_dropout (float): default=0.0
            Dropout rate for the message passing layer.
        mp_activation (str): {'relu', 'leakyrelu', 'prelu', 'tanh', 'elu'}, default='relu'
            Activation function for the message passing layer.
        aggregation (str): {'norm', 'sum', 'mean', 'attentive'}, default='norm'
            Aggregation method for the message passing layer.
            If 'attentive' is selected, the output dimension of the message passing layer will be set to `mp_dim`.
        n_layers (int): default=1
            Number of layers in the feed-forward network (FFN) after the message passing layer.
        objective (str): {'binary_crossentropy', 'mse'}, default='binary_crossentropy'
            Objective function for the MPNN model.
        predictor_dim (int): default=300
            Dimension of the hidden layers in the feed-forward network (FFN) after the message passing
        predictor_dropout (float): default=0.0
            Dropout rate for the feed-forward network (FFN) after the message passing layer.
        predictor_activation (str): {'relu', 'leakyrelu', 'prelu', 'tanh', 'elu'}, default='relu'
            Activation function for the feed-forward network (FFN) after the message passing layer.
        batch_norm (bool): default=False
            Whether to use batch normalization in the feed-forward network (FFN) after the message passing layer.
        warmup_epochs (int): default=2
            Number of warmup epochs for the learning rate scheduler.
        initial_learning_rate (float): default=1e-4
            Initial learning rate for the optimizer.
        max_learning_rate (float): default=1e-3
            Maximum learning rate for the optimizer.
        final_learning_rate (float): default=1e-4
            Final learning rate for the optimizer.
        early_stopping_params (dict): default={'mode': 'min'}
            Parameters for early stopping:
                'mode' (str): {'min', 'max'}, default='min'
                    Mode for early stopping. If 'min', the validation metric should be minimized. If 'max', the validation metric should be maximized.
                'early_stopping_rounds' (int): Number of rounds without improvement before stopping.
                'delta' (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose_eval (bool): default=True
            Whether to print evaluation metrics during training.

    Methods:
        fit(X, y, X_val=None, y_val=None, **kwargs): Fits the MPNN model to the training data. Use validation data for early stopping.
        predict(X): Predicts the target variable for the given input data.
        get_ffn_class(objective: str): Returns the appropriate feed-forward network (FFN) class based on the objective function.
        shap_explain(smiles, max_evals: int = 1000, **plot_kwargs): Plots SHAP explanations for the MPNN model predictions on the given SMILES strings.
    """
    def __init__(self,
                 binarize_labels: bool = False,
                 dealing_with_incosistency: str = 'sum',
                 epochs: int = 20,
                 batch_size: int = 64,
                 message_passing: str = 'bond',
                 mp_dim: int = 300,
                 bias: bool = False,
                 n_message_passing_iterations: int = 3,
                 messages_undirected_edges: bool = False,
                 mp_dropout: float = 0.0,
                 mp_activation: str = 'relu',
                 aggregation: str = 'norm',
                 n_layers: int = 1,
                 objective: str = 'binary_crossentropy',
                 predictor_dim: int = 300,
                 predictor_dropout: float = 0.0,
                 predictor_activation: str = 'relu',
                 batch_norm: bool = False,
                 warmup_epochs: int = 2,
                 initial_learning_rate: float = 1e-4,
                 max_learning_rate: float = 1e-3,
                 final_learning_rate: float = 1e-4,
                 early_stopping_params: dict = {'mode': 'min'},
                 verbose_eval: bool = True
                 ):
        self._init_kwargs = {
            'binarize_labels': binarize_labels,
            'dealing_with_incosistency': dealing_with_incosistency,
            'epochs': epochs,
            'batch_size': batch_size,
            'message_passing': message_passing,
            'mp_dim': mp_dim,
            'bias': bias,
            'n_message_passing_iterations': n_message_passing_iterations,
            'messages_undirected_edges': messages_undirected_edges,
            'mp_dropout': mp_dropout,
            'mp_activation': mp_activation,
            'aggregation': aggregation,
            'n_layers': n_layers,
            'objective': objective,
            'predictor_dim': predictor_dim,
            'predictor_dropout': predictor_dropout,
            'predictor_activation': predictor_activation,
            'batch_norm': batch_norm,
            'warmup_epochs': warmup_epochs,
            'initial_learning_rate': initial_learning_rate,
            'max_learning_rate': max_learning_rate,
            'final_learning_rate': final_learning_rate,
            'early_stopping_params': early_stopping_params,
            'verbose_eval': verbose_eval
        }
        self.binarize_labels = binarize_labels
        self.dealing_with_incosistency = dealing_with_incosistency
        if self.binarize_labels:
            self.label_binarizer = LabelBinarizer()
        self.epochs = epochs
        self.batch_size = batch_size
        if message_passing == 'bond':
            self.mp = BondMessagePassing(
                d_h=mp_dim,
                bias=bias,
                depth=n_message_passing_iterations,
                undirected=messages_undirected_edges,
                dropout=mp_dropout,
                activation=mp_activation
                )
        elif message_passing == 'atom':
            self.mp = AtomMessagePassing(
                d_h=mp_dim,
                bias=bias,
                depth=n_message_passing_iterations,
                undirected=messages_undirected_edges,
                dropout=mp_dropout,
                activation=mp_activation
            )
        elif message_passing == 'chemeleon':
            if not os.path.exists("chemeleon_mp.pt"):
                from urllib.request import urlretrieve
                urlretrieve(
                    r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                    "chemeleon_mp.pt",
                )
            chemeleon_mp = torch.load("chemeleon_mp.pt", weights_only=True)
            self.mp = BondMessagePassing(**chemeleon_mp['hyper_parameters'])
            self.mp.load_state_dict(chemeleon_mp['state_dict'])
            mp_dim = self.mp.output_dim
        else:
            raise ValueError("Invalid message passing type. Choose from 'bond', 'atom', 'chemeleon.")
        if aggregation == 'norm':
            self.agg = NormAggregation()
        elif aggregation == 'sum':
            self.agg = SumAggregation()
        elif aggregation == 'mean':
            self.agg = MeanAggregation()
        elif aggregation == 'attentive':
            self.agg = AttentiveAggregation(output_size=mp_dim)
        self.mp_dim = mp_dim
        self.ffn_class = self.get_ffn_class(objective)
        self.predictor_dim = predictor_dim
        self.predictor_dropout = predictor_dropout
        self.predictor_activation = predictor_activation
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.warmup_epochs = warmup_epochs
        self.initial_learning_rate = initial_learning_rate
        self.max_learning_rate = max_learning_rate
        self.final_learning_rate = final_learning_rate
        self.early_stopping_params = early_stopping_params
        self.verbose_eval = verbose_eval
        self.model = None
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def fit(self, X: np.ndarray, y: np.ndarray, X_val:np.ndarray|None=None, y_val:np.ndarray|None=None, **kwargs):
        sample_weight = self.calculate_sample_weight(y)
        if self.binarize_labels:
            y = self.label_binarizer.binarize_labels(y)
            y_val = self.label_binarizer.binarize_labels(y_val) if y_val is not None else None
        if len(y.shape) == 1:
            output_dim = 1
            y = y.reshape(-1, 1) # so y in shape (n_samples, output_dim)
            y_val = y_val.reshape(-1, 1) if y_val is not None else None
        else:
            output_dim = y.shape[1]
        ffn = self.ffn_class(
            n_tasks=output_dim,
            input_dim=self.mp_dim,
            hidden_dim=self.predictor_dim,
            n_layers=self.n_layers,
            dropout=self.predictor_dropout,
            activation=self.predictor_activation,
            )
        self.model = MPNN(
            self.mp,
            self.agg,
            ffn,
            # metrics=['accuracy', 'f1'],
            batch_norm=self.batch_norm,
            warmup_epochs=self.warmup_epochs,
            init_lr=self.initial_learning_rate,
            max_lr=self.max_learning_rate,
            final_lr=self.final_learning_rate,
            )
        train_datapoints = [MoleculeDatapoint.from_smi(smiles, y_single, weight=weight) for smiles, y_single, weight in zip(X, y, sample_weight)]
        val_datapoints = [MoleculeDatapoint.from_smi(smiles, y_single) for smiles, y_single in zip(X_val, y_val)] if X_val is not None and y_val is not None else []
        train_dataset = MoleculeDataset(train_datapoints, featurizer=self.featurizer)
        val_dataset = MoleculeDataset(val_datapoints, self.featurizer) if X_val is not None and y_val is not None else None
        train_dataloader = build_dataloader(train_dataset, batch_size=self.batch_size)
        val_dataloader = build_dataloader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset is not None else None
        if X_val is not None and y_val is not None:
            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_params.get('early_stopping_rounds', 20),
                mode=self.early_stopping_params.get('mode', 'min'),
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:6]  
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filename=f'{timestamp}_{unique_id}' + '{epoch:02d}-{step:02d}',
                monitor='val_loss',
                mode=self.early_stopping_params.get('mode', 'min'),
                save_top_k=1,
            )
            callbacks = [early_stopping_callback, checkpoint_callback]
        else:
            callbacks = []
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
            enable_progress_bar=self.verbose_eval,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=self.epochs, # number of epochs to train for
            callbacks=callbacks, 
            enable_model_summary=self.verbose_eval,
            )
        best_epoch = None
        if val_dataloader is not None:
            trainer.fit(self.model, train_dataloader, val_dataloader)
            best_model_path = trainer.checkpoint_callback.best_model_path
            checkpoint = torch.load(best_model_path, weights_only=False)
            best_epoch = checkpoint['epoch']
            os.remove(best_model_path)  # Remove the checkpoint file to save space
        else:
            trainer.fit(self.model, train_dataloader)
        return best_epoch
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        data_points = [MoleculeDatapoint.from_smi(smiles) for smiles in X]
        dataset = MoleculeDataset(data_points, featurizer=self.featurizer)
        dataloader = build_dataloader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=False,
                enable_progress_bar=self.verbose_eval,
                enable_model_summary=self.verbose_eval,
                accelerator="cuda" if torch.cuda.is_available() else "cpu",
                devices=1,
            )
            predictions = trainer.predict(self.model, dataloader)
            predictions = np.concatenate(predictions, axis=0)
            if len(predictions.shape) > 1:
                predictions = np.squeeze(predictions)
        if self.binarize_labels:
            if len(np.squeeze(predictions).shape) == 1:
                one_sample = True
            else:
                one_sample = False
            predictions = self.label_binarizer.inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency, one_sample=one_sample)
        return predictions
    
    def predict_from_fingerprint(self, X_fp: np.ndarray) -> np.ndarray:
        X_fp = torch.tensor(X_fp, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model.predictor(X_fp)
        predictions = predictions.cpu().numpy()
        if self.binarize_labels:
            if len(np.squeeze(predictions).shape) == 1:
                one_sample = True
            else:
                one_sample = False
            predictions = self.label_binarizer.inverse_binarize_labels(predictions, dealing_with_incosistency=self.dealing_with_incosistency, one_sample=one_sample)
        return predictions
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Generates molecular embeddings using the MPNN model.

        Parameters:
            X (np.ndarray): Array of SMILES strings representing the molecules.

        Returns:
            np.ndarray: Array of molecular embeddings.
        """
        data_points = [MoleculeDatapoint.from_smi(smiles) for smiles in X]
        dataset = MoleculeDataset(data_points, featurizer=self.featurizer)
        dataloader = build_dataloader(dataset, batch_size=self.batch_size, shuffle=False)
        embeddings = []
        for batch in dataloader:
            batch_embeddings = self.model.fingerprint(batch.bmg, batch.V_d, batch.X_d)
            embeddings.append(batch_embeddings.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
        
    def get_ffn_class(self, objective):
        if objective == 'binary_crossentropy':
            return BinaryClassificationFFN
        elif objective == 'crossentropy':
            return MulticlassClassificationFFN
        elif objective == 'mse':
            return RegressionFFN
        else:
            raise ValueError(f"Unknown objective: {objective}. Available objectives: binary_crossentropy, crossentropy, mse, mae.")
    
    def save(self, path: str):
        """Saves the model to the specified path."""
        from chemprop.models.utils import save_model
        save_model(path, self.model)

    def load(self, path: str):
        """Loads the model from the specified path."""
        self.model = MPNN.load_from_file(path)


class ChemeleonPredictor(ChemPropMPNNPredictor):
    """Chemeleon MPNN Predictor for molecular property prediction."""
    def __init__(self, *args, **kwargs):
        kwargs['message_passing'] = 'chemeleon'
        super().__init__(*args, **kwargs)