import json
import pickle
from typing import Optional, Callable
import logging

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split, StratifiedKFold

from ...estimation import CalibratedHGBCEstimatorFactory, BaseEstimatorFactory, BaseEstimator
from ...feature_selection.inferred_selector import BaseFeatureSelector
from ...feature_selection.inferred_selector import InferredFeatureSelector
from ...modeling.models.binary_model import BinaryModel
from ...postprocessing import BasePostprocessor
from ...utils.io import ModelPaths, NumpyEncoder, NumpyDecoder

logger = logging.getLogger(__name__)


class BinaryModelFactory:
    """Factory class for creating and fitting BinaryModel objects.

    Combines a FeatureSelector, EstimatorFactory, and Postprocessor into a BinaryModel and trains it. Primarily acts as a
    lightweight wrapper for


    Parameters
    ----------
    feature_selector :
    estimator_factory :
    postprocessor :
    n_optimization_rounds :
    n_splits :
    test_size :
    scoring :
    random_state :
    """

    estimator_factory: BaseEstimatorFactory
    _model: BinaryModel

    test_size: float

    cv: BaseCrossValidator
    scoring: Optional[list[str | Callable]]
    random_state: Optional[int]
    model_report: dict

    def __init__(self,
                 feature_selector: Optional[BaseFeatureSelector] = None,
                 estimator_factory: Optional[BaseEstimatorFactory] = None,
                 postprocessor: Optional[BasePostprocessor] = None,
                 n_optimization_rounds=10,
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 scoring: str = 'roc_auc',
                 random_state: Optional[int] = None):

        feature_selector = feature_selector or InferredFeatureSelector()

        # We need a random state to replicate cross validation folds. If not provided one, we randomly choose one
        self.random_state = np.random.randint(0, 1e9) if random_state is None else random_state
        self.n_splits = n_splits
        self.scoring = scoring
        self.n_optimization_rounds = n_optimization_rounds

        self._model = BinaryModel.__new__(BinaryModel)
        self._model.feature_selector = feature_selector
        self._model.random_state = random_state
        self._model.postprocessor = postprocessor
        self._model.estimator = None

        self.test_size = test_size

        self.model_report = {}

        self.estimator_factory = estimator_factory or CalibratedHGBCEstimatorFactory(cv=self.cv, scoring=scoring, logging_callback=None)

    @property
    def model(self) -> Optional[BinaryModel]:
        """Returns the model if it has been built, otherwise return None

        Returns
        -------
        BinaryModel if model has been fit, otherwise None
        """
        if self._model.estimator is None:
            return None
        else:
            return self._model

    @property
    def cv(self) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    @property
    def feature_selector(self) -> BaseFeatureSelector:
        return self._model.feature_selector if self.is_fitted() else None

    @property
    def estimator(self) -> BaseEstimator:
        return self._model.estimator if self.is_fitted() else None

    @property
    def postprocessor(self) -> BasePostprocessor:
        return self._model.postprocessor if self.is_fitted() else None

    def is_fitted(self) -> bool:
        return self.model is not None

    def _fit(self, X: pd.DataFrame, y: pd.Series, optimize: bool) -> (pd.DataFrame, pd.Series):
        # Reset _model report
        self.model_report = {}

        logger.info(f"Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        logger.info(f"Train data has {len(X_train)} records and Test data has {len(X_test)} records.")

        logger.info(f"Begin fitting {self._model.feature_selector.__class__.__name__} to training data.")
        X_train = self._model.feature_selector.fit_transform(X_train, y_train)
        logger.info(f"Finished fitting {self._model.feature_selector.__class__.__name__}.")

        self.estimator_factory.set_params(cv=self.cv,
                                          #random_state=self.random_state,
                                          scoring=self.scoring,
                                          n_optimization_rounds=self.n_optimization_rounds,
                                          logging_callback=self.create_logging_callback('estimator_factory'))

        if optimize and self.n_optimization_rounds:
            logger.info(f"Begin optimizing estimator to training data using {self.estimator_factory.__class__.__name__}")
            estimator = self.estimator_factory.optimize(X_train, y_train, self._model.feature_selector)
            logger.info(f"Finished optimizing estimator.")
        else:
            logger.info(f"Begin fitting estimator to training data using {self.estimator_factory.__class__.__name__}")
            estimator = self.estimator_factory.fit(X_train, y_train, self._model.feature_selector)
            logger.info(f"Finished fitting estimator.")

        self._model = BinaryModel(feature_selector=self._model.feature_selector,
                                  estimator=estimator,
                                  postprocessor=self._model.postprocessor,
                                  random_state=self.random_state)

        if self._model.postprocessor is not None:
            logger.info(f"Begin fitting {self._model.postprocessor.__class__.__name__} to data.")
            self._model._fit_postprocessor(X_train, y_train, self.cv)
            logger.info(f"Finished fitting {self._model.postprocessor.__class__.__name__} to data.")

        return self._model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BinaryModel:
        return self._fit(X, y, False)

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> BinaryModel:
        return self._fit(X, y, True)

    def train_test_split(self, X: pd.DataFrame, y: pd.Series) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
        # Ensures we get the same train test split everytime.
        y = y.astype(int)
        return train_test_split(X, y, test_size=self.test_size, stratify=y,
                                random_state=self.random_state)

    def create_logging_callback(self, key: str) -> Callable:
        def log_message_dict(message_dict: dict):
            if key not in self.model_report:
                self.model_report[key] = {}
            self.model_report[key].update(message_dict)

        return log_message_dict

    def save(self, model_paths: ModelPaths):
        metadata = self.__getstate__()
        model = metadata.pop("_model")

        model_report = metadata.pop("model_report")
        estimator_factory = metadata.pop("estimator_factory")

        model.save(model_paths)

        with open(model_paths.estimator_factory, 'wb') as f:
            pickle.dump(estimator_factory, f)

        with open(model_paths.model_factory, 'w') as f:
            json.dump(metadata, f, cls=NumpyEncoder)

    @classmethod
    def load(cls, model_paths: ModelPaths):
        self = cls()

        self._model = BinaryModel.load(model_paths)

        with open(model_paths.model_factory, 'r') as f:
            self.__dict__.update(json.load(f, cls=NumpyDecoder))

        with open(model_paths.estimator_factory, 'rb') as f:
            self.estimator_factory = pickle.load(f)
            self.estimator_factory.set_params(logging_callback=self.create_logging_callback('estimator_factory'))

        return self

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        # Add an extra tab to every line for nested indentation. Makes reading easier.
        estimator_factory_repr = '\t'.join(self.estimator_factory.__repr__().split('\n'))
        model_repr = '\n\t'.join(self._model.__repr__().split('\n'))
        return f"BinaryModelFactory(\n" \
               f"\testimator_factory: {estimator_factory_repr}\n" \
               f"\tmodel: {model_repr}\n" \
               f")"

