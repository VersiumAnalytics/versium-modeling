import sys
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

MAX_CARDINALITY = 255
MAX_RELATIVE_CARDINALITY = .2



class BaseFeatureSelector(ABC, TransformerMixin):
    # init args
    include_columns: list[str]
    exclude_columns: list[str]

    # internal vars
    columns: list[str]
    numeric: set[str]
    categorical: set[str]
    integers: set[str]
    strings: set[str]
    datetime: set[str]

    def __init__(self, include_columns: list[str] = (), exclude_columns: list[str] = ()):
        self.include_columns = include_columns
        self.exclude_columns = exclude_columns
        self.columns = []
        self.numeric = set()
        self.categorical = set()
        self.integers = set()
        self.strings = set()
        self.datetime = set()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        pass

    def get_numeric_features(self) -> list[str]:
        return [c for c in self.columns if c in self.numeric]

    def get_categorical_features(self) -> list[str]:
        return [c for c in self.columns if c in self.categorical]

    def get_integer_features(self) -> list[str]:
        return [c for c in self.columns if c in self.integers]

    def get_string_features(self) -> list[str]:
        return [c for c in self.columns if c in (self.categorical - self.integers)]

    def get_numeric_feature_mask(self) -> np.ndarray:
        return np.array([c in self.numeric for c in self.columns], dtype=bool)

    def get_integer_feature_mask(self) -> np.ndarray:
        return np.array([c in self.integers for c in self.columns], dtype=bool)

    def get_categorical_feature_mask(self) -> np.ndarray:
        return np.array([c in self.categorical for c in self.columns], dtype=bool)

    def get_string_feature_mask(self) -> np.ndarray:
        string_set = self.categorical - self.integers
        return np.array([c in string_set for c in self.columns], dtype=bool)

    @staticmethod
    def load(state_dict):
        classname = state_dict.pop('<__classname__>')
        cls = getattr(sys.modules[__name__], classname)
        instance = object.__new__(cls)
        instance.__setstate__(state_dict)
        return instance

    def __getstate__(self):
        state = self.__dict__.copy()
        state['<__classname__>'] = self.__class__.__name__
        for key in ('numeric', 'categorical', 'integers', 'strings', 'datetime'):
            state[key] = list(state[key])

        return state

    def __setstate__(self, state):
        state.pop('<__classname__>', None)
        for key in ('numeric', 'categorical', 'integers', 'strings', 'datetime'):
            state[key] = set(state[key])

        self.__dict__.update(state)