import logging
from functools import partial
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder
from skopt import gp_minimize
from skopt.space import Integer, Real

from .base_estimator_factory import BaseEstimatorFactory
from ..estimators import Pipeline
from ...feature_selection import BaseFeatureSelector

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {'estimator__base_estimator__learning_rate': 0.01, 'estimator__base_estimator__max_depth': 512,
                  'estimator__base_estimator__max_leaf_nodes': 50, 'estimator__base_estimator__min_samples_leaf': 2,
                  'estimator__base_estimator__max_iter': 125}

MAX_NUM_RANDOM_STARTS = 10


class CalibratedHGBCEstimatorFactory(BaseEstimatorFactory):

    def __init__(self, cv=5, scoring='roc_auc', n_optimization_rounds=20, logging_callback=None, cachedir: Optional[str] = None,
                 verbose_feature_names_out: bool = False, estimator_params=None):
        super().__init__()
        self.cachedir = cachedir
        self.scoring = scoring

        self.n_optimization_rounds = n_optimization_rounds
        self.cv = cv

        self.verbose_feature_names = verbose_feature_names_out
        self.logging_callback = logging_callback
        self.estimator_params = DEFAULT_PARAMS if estimator_params is None else estimator_params

    def __call__(self, feature_selector: Optional[BaseFeatureSelector] = None):
        cat_selector = make_column_selector(
            dtype_include=object) if feature_selector is None else feature_selector.get_categorical_features()
        num_selector = make_column_selector(dtype_include=float) if feature_selector is None else feature_selector.get_numeric_features()

        X_trans = ColumnTransformer([('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan),
                                      cat_selector),
                                     ('numerical', 'passthrough',
                                      num_selector)
                                     ], verbose_feature_names_out=self.verbose_feature_names)

        cat_features = () if feature_selector is None else range(len(feature_selector.get_categorical_features()))
        base_estimator = HistGradientBoostingClassifier(categorical_features=cat_features, scoring='roc_auc')
        estimator = CalibratedClassifierCV(base_estimator, cv=5, ensemble=True)
        pipeline = Pipeline(steps=[
            ("X_transformer", X_trans),
            ("estimator", estimator),
        ], memory=self.cachedir)

        pipeline.set_params(**self.estimator_params)
        return pipeline

    @property
    def param_grid(self):
        return {
            'estimator__base_estimator__learning_rate': Real(0.01, 1.0, 'uniform', name='learning_rate'),
            'estimator__base_estimator__max_depth': Integer(1, 512, 'log-uniform', base=2, name='max_depth'),
            'estimator__base_estimator__max_leaf_nodes': Integer(2, 128, 'log-uniform', base=2, name='max_leaf_nodes'),
            'estimator__base_estimator__min_samples_leaf': Integer(2, 1024, 'log-uniform', base=2, name='min_samples_leaf'),
            'estimator__base_estimator__max_iter': Integer(50, 125, 'uniform', name='max_iter')
        }

    def _objective(self, X: pd.DataFrame, y: pd.Series, model, param_names: list[str], params: list[Any]):
        param_dict_orig = dict(zip(param_names, params))
        param_dict = param_dict_orig.copy()
        param_dict['estimator__base_estimator__scoring'] = self.scoring
        model.set_params(**param_dict)

        scores = cross_validate(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1, return_train_score=True)

        test_score = np.mean(scores['test_score'])
        train_score = np.mean(scores['train_score'])
        msg = f"Optimizing {self.scoring} scores:\ttrain: {train_score}\ttest: {test_score}\tparams: {param_dict_orig}"

        logger.info(msg)
        return -1 * test_score

    def log_dict(self, message_dict: dict):
        if self.logging_callback is None:
            return

        self.logging_callback(message_dict)

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_selector: BaseFeatureSelector):
        logger.info(f"Begin fitting estimator.")
        model = self(feature_selector)
        model.fit(X, y)
        logger.info(f"Finished fitting estimator.")
        return model

    def optimize(self, X: pd.DataFrame, y: pd.Series, feature_selector: BaseFeatureSelector):
        logger.info(f"Begin hyperparameter optimization for estimator. Optimizing metric {self.scoring}")
        model = self(feature_selector)
        objective = partial(self._objective, X, y, model, self.param_grid.keys())
        n_calls = self.n_optimization_rounds
        n_random_starts = min(MAX_NUM_RANDOM_STARTS, n_calls // 2)
        res_gp = gp_minimize(objective, dimensions=self.param_grid.values(), n_random_starts=n_random_starts, n_calls=n_calls)

        # Get the optimized params from results.
        idx_min = res_gp.func_vals.argmin()
        best_params = dict(zip(self.param_grid.keys(), res_gp.x))
        model.set_params(**best_params)
        model.fit(X, y)
        logger.info(f"Finished hyperparameter optimization for estimator.")
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('logging_callback')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logging_callback = None
