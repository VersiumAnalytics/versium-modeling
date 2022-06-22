from abc import ABC, abstractmethod

import pandas as pd

from ..estimators import BaseEstimator
from ...feature_selection import BaseFeatureSelector


class BaseEstimatorFactory(ABC):

    def __init__(self):
        pass

    def set_params(self, **kwargs):
        allowed_params = set(vars(self).keys())
        for k, v in kwargs.items():
            if k not in allowed_params:
                raise AttributeError(f"No attribute named `{k}` in {self.__class__.__name__}")
            setattr(self, k, v)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, feature_selector: BaseFeatureSelector) -> BaseEstimator:
        pass

    @abstractmethod
    def optimize(self, X: pd.DataFrame, y: pd.Series, feature_selector: BaseFeatureSelector) -> BaseEstimator:
        pass
