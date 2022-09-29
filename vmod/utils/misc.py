"""misc.py: Miscellaneous utilities."""
import inspect
from typing import Callable


def filter_params(d: dict, func: Callable) -> dict:
    """Filter a dictionary with respect to parameters of a given function.


    Parameters
    ----------
    d : dict
        Dictionary to be filtered.
    func : callable
        Function or class with parameters to filter.

    Returns
    -------
    filtered_dict : dict
        Filtered dictionary.
    """

    signature = inspect.signature(func)
    filter_keys = [param.name for param in signature.parameters.values() if param.kind in
                   (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)]
    filtered_dict = {filter_key: d.get(filter_key, None) for filter_key in filter_keys if filter_key in d}
    return filtered_dict
