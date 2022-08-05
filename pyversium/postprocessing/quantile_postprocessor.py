from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import QuantileTransformer

from .base_postprocessor import BasePostprocessor, PredictMethodName


class QuantilePostprocessor(QuantileTransformer, BasePostprocessor):

    def __init__(self,
                 *,
                 connector_method_name: PredictMethodName = 'predict_proba',
                 **kwargs):
        """
        Postprocessor that takes in scores and applies a quantile transformation over them. By default transforms the output into a
        uniform integer distribution from 0-100. Can also be used to normally distribute scores by using the
        `output_distribution='normal' keyword argument.

        Parameters
        ----------
        connector_method_name : str
            The name of the method to use for retrieving predictions from an estimator
        **kwargs: kwargs
            Keyword arguments to pass to the constructor of QuantileTransformerbsampling procedure may differ for value-identical sparse and dense matrices.
        """
        self.connector_method_name = connector_method_name

        super().__init__(**kwargs)

    def transform(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        X = super().transform(X)
        return np.round(X*100, 0)

    def __repr__(self, N_CHAR_MAX=700):  # Include N_CHAR_MAX to match signature of QuantileTransformer
        return f"{self.__class__.__name__}(n_quantiles={self.n_quantiles}, output_distribution={self.output_distribution}," \
               f" subsample={self.subsample}, random_state={self.random_state})"
