"""base_postprocessor.py: Defines base class for postprocessors that transform scores from a model."""
from abc import abstractmethod
from typing import Literal
from typing import Optional

from numpy.typing import ArrayLike

PredictMethodName = Literal["predict", 'decision_function', "predict_proba", "predict_log_proba"]


class BasePostprocessor:
    connector_method_name: PredictMethodName

    def __init__(self, connector_method_name: PredictMethodName, *args, **kwargs):
        self.connector_method_name = connector_method_name
        super().__init__(*args, **kwargs)

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        pass

    @abstractmethod
    def transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:
        pass

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:
        return self.fit(X, y).transform(X, y)