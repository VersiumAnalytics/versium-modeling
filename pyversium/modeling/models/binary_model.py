import pickle
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
import logging
from ...feature_selection.inferred_selector import BaseFeatureSelector
from ...postprocessing import BasePostprocessor
from ...utils.io import ModelPaths

logger = logging.getLogger(__name__)

def _get_prediction_method(clf):
    """Return prediction method.

    `decision_function` method of `clf` returned, if it
    exists, otherwise `predict_proba` method returned.

    Parameters
    ----------
    clf : Estimator instance
        Fitted classifier to obtain the prediction method from.

    Returns
    -------
    prediction_method : callable
        The prediction method.
    method_name : str
        The name of the prediction method.
    """
    if hasattr(clf, "transform"):
        method = getattr(clf, "transform")
        return method, "transform"
    elif hasattr(clf, "predict_proba"):
        method = getattr(clf, "predict_proba")
        return method, "predict_proba"
    elif hasattr(clf, "decision_function"):
        method = getattr(clf, "decision_function")
        return method, "decision_function"
    else:
        raise RuntimeError(
            "'estimator' has no 'decision_function' or 'predict_proba' method."
        )


class BinaryModel:
    feature_selector: BaseFeatureSelector
    estimator: BaseEstimator
    postprocessor: Optional[BasePostprocessor]
    random_state: np.int

    def __init__(self, feature_selector: BaseFeatureSelector,
                 estimator: BaseEstimator,
                 postprocessor: Optional[BasePostprocessor] = None,
                 random_state: Optional[int|np.int] = None):
        """
        A Binary Classification Model with 3 stages:
            A FeatureSelector to select the model fields from the data and their type
            A classifier that maps input data to predicted classes
            An optional postprocessor that takes the output from estimator and applies a transformation (such as score normalization)

        Parameters
        ----------
        feature_selector : FeatureSelector instance
            A FeatureSelector to select fields from input data.
        estimator : BaseEstimator
            A scikit-learn Estimator, Pipeline, or a custom created class that derives from BaseEstimator
        postprocessor : Postprocessor instance
            A postprocessor that takes the output from estimator and transforms it
        random_state : int
            Integer used as random seed
        """
        if postprocessor is not None and not hasattr(estimator, postprocessor.connector_method_name):
            raise AttributeError(f"{type(estimator)} has no method named `{postprocessor.connector_method_name}`."
                                 "Check `postprocessor.connector_method_name`.")
        self.feature_selector = feature_selector
        self.estimator = estimator
        self.postprocessor = postprocessor
        self.random_state = np.random.randint(0, 1e9) if random_state is None else np.int(random_state)
        logger.debug(f"Creating {self.__class__.__name__} with"
                     f" feature_selector: {feature_selector.__class__.__name__}"
                     f" estimator: {estimator.__class__.__name__}"
                     f" postprocessor: {postprocessor.__class__.__name__}"
                     f" random_state: {random_state}")

    def fit(self, X: pd.DataFrame, y: pd.Series = None, cv=None):
        logger.debug("Fitting feature selector.")
        X = self.feature_selector.fit_transform(X, y)

        logger.debug("Fitting estimator.")
        self.estimator.fit(X, y)

        if self.postprocessor is not None:
            logger.debug("Fitting postprocessor.")
            self._fit_postprocessor(X, y, cv)

        return self

    def _fit_postprocessor(self, X, y, cv=None):
        if self.postprocessor is None:
            return self

        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        scores = cross_val_predict(self.estimator, X, y,
                                   cv=cv,
                                   method=self.postprocessor.connector_method_name,
                                   n_jobs=-1)
        scores = scores[:, 1].reshape(-1, 1)
        test_splits = list(cv.split(X, y))
        test_indices = np.concatenate([test for _, test in test_splits])
        y_true = y.iloc[test_indices]
        self.postprocessor.fit(scores, y_true)

    def predict(self, X: pd.DataFrame, y: pd.Series = None):
        X = self.feature_selector.transform(X, y)
        return self.estimator.predict(X)

    def decision_function(self, X: pd.DataFrame, y: pd.Series = None):
        X = self.feature_selector.transform(X, y)
        return self.estimator.decision_function(X)

    def predict_log_proba(self, X: pd.DataFrame, y: pd.Series = None):
        X = self.feature_selector.transform(X, y)
        return self.estimator.predict_log_proba(X)

    def predict_proba(self, X: pd.DataFrame, y: pd.Series = None):
        X = self.feature_selector.transform(X, y)
        return self.estimator.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series = None):
        """ Alias for transform
        """
        return self.transform(X, y)

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        X = self.feature_selector.transform(X, y)

        if self.postprocessor is None:
            if isinstance(self.estimator, Pipeline):
                # estimator will be last step in pipeline. Otherwise Pipeline implements all prediction methods and passes
                # that on to the estimator
                _, method_name = _get_prediction_method(self.estimator[-1])
                pred_method = getattr(self.estimator, method_name)
            else:
                pred_method, method_name = _get_prediction_method(self.estimator)

            return pred_method(X)
        else:
            pred_method, method_name = getattr(self.estimator, self.postprocessor.connector_method_name),\
                                       self.postprocessor.connector_method_name

        scores = pred_method(X)
        return self.postprocessor.transform(scores[:, 1].reshape(-1, 1)).flatten()

    def __repr__(self):
        return f"BinaryModel(\n" \
               f"\tfeature_selector: {self.feature_selector.__repr__()}\n" \
               f"\testimator: {self.estimator.__repr__()}\n" \
               f"\tpostprocessor: {self.postprocessor.__repr__()}" \
               f")"

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, model_paths: ModelPaths):
        with open(model_paths.model_file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_paths: ModelPaths):
        with open(model_paths.model_file, 'rb') as f:
            self = pickle.load(f)
        return self
