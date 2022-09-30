"""response_handler.py: Defines functions for handling different http responses."""
import logging
from typing import Callable
logger = logging.getLogger(__name__)

ResponseHandlerType = Callable[[dict, str, ...], dict | list]


def default_response_handler(response: dict, url: str, **kwargs) -> dict | list:
    """Default response handler. Does no processing, just returns the response as given.

        Parameters
        -------
        response : dict
            json response returned from request.

        url : str
            Url that the response is from.

        **kwargs : kwargs
            Additional arguments that may be needed to

        Returns
        -------
        dict
    """
    return response


def api_versium_com(response: dict, url: str, **kwargs) -> dict:
    raw_json = kwargs.pop('raw_json', False)
    if "Versium" in response:
        response["versium"] = response.pop("Versium")

    if raw_json:
        data_dict = response['versium']
    elif response["versium"].get('results', []) and isinstance(response['versium']['results'][0], dict):
        data_dict = response['versium']['results'][0]
    else:
        data_dict = {}

    return data_dict
