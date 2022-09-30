"""hgbc_factory.py: Defines a factory for creating calibrated histogram gradient boosting classifier pipelines."""
import logging
from functools import partial
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, BaseCrossValidator
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
    """
    Factory class for creating and optimizing histogram gradient boosting classifier pipelines with calibrated probability scores.

    Parameters
    ----------
    cv : BaseCrossValidator or int
        Number of cross validation folds to use for calibration and optimization. Alternatively, a scikit-learn cross validator.

    scoring : str
        Scoring metric to use for optimization. Uses scikit-learn scoring metric strings. For list of all possible scoring metrics see
        https://scikit-learn.org/stable/modules/model_evaluation.html

    n_optimization_rounds : int
        Number of different hyperparameter combinations to try when optimizing.

    cachedir : str
        Directory to use for caching column transformations. This helps speed up optimization by removing the need to recalculate column
        transformations such as categorical encoding for every round of hyperparameter tuning.

    verbose_feature_names : bool
        If true, all feature names are prefixed with the column transformer that generated them.

    estimator_params : dict[str, Any]
        Dictionary of parameter-name, value pairs to set the estimator to. Leave this as None unless you know the structure of the pipeline.
        Since the pipeline produced by this factory is a composition of nested estimators, parameters need to be accessed via the dedicated
        <estimator>__<parameter> syntax for nested params used by scikit-learn. For example, the Pipeline generated is composed of
        steps: `X_transformer`, `estimator`. The `estimator` step of the pipeline is a CalibratedClassifier with a HistGradientBoostingClassifier
        as its `base_estimator`. Thus, if you want to change the learning rate of the gradient boosting classifier, you need to specify the
        parameter as `estimator__base_estimator__learning_rate`. Likewise, if you want to change the dtype of the categorical encoder, you
        need to specify the parameter as `X_transformer__categorical__dtype`. For the structure of the generated pipeline, see the __call__
        method of this class.
        For further help see https://scikit-learn.org/stable/modules/grid_search.html#composite-estimators-and-parameter-spaces.
    """

    cv: int
    scoring: str
    n_optimization_rounds: int
    cachedir: Optional[str]
    verbose_feature_names: bool
    estimator_params: dict[str, Any]

    def __init__(self,
                 cv: BaseCrossValidator | int = 5,
                 scoring: str = 'roc_auc',
                 n_optimization_rounds: int = 20,
                 cachedir: Optional[str] = None,
                 verbose_feature_names: bool = False,
                 estimator_params: Optional[dict[str, Any]] = None):

        super().__init__()
        self.cachedir = cachedir
        self.scoring = scoring

        self.n_optimization_rounds = n_optimization_rounds
        self.cv = cv

        self.verbose_feature_names = verbose_feature_names
        self.estimator_params = DEFAULT_PARAMS if estimator_params is None else estimator_params

    def __call__(self, feature_selector: Optional[BaseFeatureSelector] = None):

        cat_selector = make_column_selector(dtype_include=object) if feature_selector is None else feature_selector.get_categorical_features()
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
        """Objective function for hyperparameter optimization."""
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
        best_params = dict(zip(self.param_grid.keys(), res_gp.x))
        model.set_params(**best_params)
        model.fit(X, y)
        logger.info(f"Finished hyperparameter optimization for estimator.")
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
