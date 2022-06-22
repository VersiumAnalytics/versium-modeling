from typing import Optional
import logging
import numpy as np
import pandas as pd
import pandas._libs.lib as lib
import pandas._libs.ops as libops
import pandas._libs.parsers as parsers
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ..constants import STR_NA_VALUES
from .base_selector import BaseFeatureSelector

MAX_CARDINALITY = 255
MAX_RELATIVE_CARDINALITY = .2

logger = logging.getLogger(__name__)

def _infer_types(values, na_values=STR_NA_VALUES, try_num=True, try_bool=True):
    """
    Infer types of values, possibly casting

    Parameters
    ----------
    values : ndarray
    na_values : set
    try_num_bool : bool, default try
       try to cast values to numeric (first preference) or boolean

    Returns
    -------
    converted : ndarray
    na_count : int
    """
    if isinstance(values, pd.Series):
        values = values.values
    result = values

    if try_num and is_object_dtype(result):
        # exclude e.g DatetimeIndex here
        try:
            result, null_mask = lib.maybe_convert_numeric(values, na_values, False, convert_to_masked_nullable=True)
        except (ValueError, TypeError):
            # e.g. encountering datetime string gets ValueError
            #  TypeError can be raised in floatify
            pass
        else:
            if is_integer_dtype(result):
                result = pd.Series(result, dtype='Int64')
                if null_mask is not None:
                    result[null_mask] = pd.NA
    else:
        if result.dtype == np.object_:
            na_count = parsers.sanitize_objects(result, na_values)

    if try_bool and is_object_dtype(result):
        try:
            result, null_mask = libops.maybe_convert_bool(
                np.asarray(result),
                true_values=None,
                false_values=None,
                convert_to_masked_nullable=True,
            )
        #except (ValueError, TypeError):
        except Exception as e:
            # e.g. encountering datetime string gets ValueError
            #  TypeError can be raised in floatify
            pass
        else:
            if is_bool_dtype(result):
                result = pd.Series(result, dtype='boolean')
                if null_mask is not None:
                    result[null_mask] = pd.NA

                return result

    #return result, na_count
    return result


class InferredFeatureSelector(BaseFeatureSelector):
    column_dtypes: Optional[dict[str, str]]
    num_sampled: Optional[int]
    fill_rates: dict[str, int]
    fill_rate_threshold: float

    def __init__(self,
                 include_columns: list[str] = (),
                 exclude_columns: list[str] = (),
                 column_dtypes: Optional[dict[str, str]] = None,
                 fill_rate_threshold=0.6):
        super().__init__()

        self.include_columns = include_columns
        self.exclude_columns = exclude_columns
        self.column_dtypes = column_dtypes
        self.fill_rates = {}
        self.fill_rate_threshold = fill_rate_threshold
        self.num_unique = None
        self.num_sampled = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):

        if not self.include_columns:
            include = X.columns
        else:
            include = self.include_columns

        exclude_set = set(self.exclude_columns)
        if y is not None:
            exclude_set.add(y.name)

        # Doing list comprehension instead of set difference preserves order of include.
        model_cols = [i for i in include if i not in exclude_set]
        if set(model_cols) != set(model_cols).intersection(set(X.columns)):
            raise ValueError("The following columns were supplied to `include_columns` but were not found in the dataset:"
                             f"{set(model_cols) - set(X.columns)}")


        str_lens = X[model_cols].apply(lambda x: x.str.len())
        fixed_width = (str_lens.max() == str_lens.min())
        del str_lens

        # Booleans are somewhat annoying in that they will be inferred by lib.maybe_convert_bool but Pandas will throw an error when
        # trying to cast as "boolean" using astype() function. So we avoid them altogether and just treat them as strings.
        num_unique = X[model_cols].nunique(dropna=False)
        X_inferred = X[model_cols].apply(_infer_types, axis=0, raw=False, try_bool=False)
        fill_rate = X_inferred.notna().mean()

        dtypes = X_inferred.dtypes
        integers = dtypes.apply(is_integer_dtype)
        strings = dtypes.apply(is_string_dtype)

        high_cardinality = (num_unique / len(X_inferred) > MAX_RELATIVE_CARDINALITY) | (num_unique > MAX_CARDINALITY)
        no_variance = num_unique < 2
        bad_fill_rate = fill_rate < self.fill_rate_threshold
        bad_quality = no_variance | bad_fill_rate

        categorical = (strings | (integers & fixed_width)) & ~high_cardinality & ~bad_quality
        integer_ids = integers & fixed_width & high_cardinality & ~bad_quality
        numerics = dtypes.apply(is_numeric_dtype) & ~categorical & ~integer_ids & ~bad_quality

        string_features = set(X_inferred.columns[strings])
        numeric_features = set(X_inferred.columns[numerics])
        categorical_features = set(X_inferred.columns[categorical])
        feature_cols = numeric_features | categorical_features

        self.columns = [c for c in model_cols if c in feature_cols]
        self.numeric = numeric_features
        self.categorical = categorical_features
        self.integers = set(X_inferred.columns[integers])
        self.strings = string_features
        self.num_sampled = len(X_inferred)

        # Convert dtypes to dict for only columns we need, and convert each dtype to its string representation
        self.column_dtypes = {k: str(v) for k, v in dtypes[self.columns].to_dict().items()}

        self.num_unique = num_unique[self.columns].to_dict()

        if not self.columns:
            raise ValueError("No features left after performing feature selection. Try providing additional features")

        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        # For now scikit-learn doesn't support Pandas nullable types, so we need to convert "Int64" categorical types to strings and "Int64"
        # numeric types to float64

        column_dtype_remap = {k: ("object" if k in self.categorical else "float64") for k in self.column_dtypes.keys()}
        return X[self.columns].astype(column_dtype_remap)

