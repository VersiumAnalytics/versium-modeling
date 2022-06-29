import logging
logger = logging.getLogger(__name__)


def default_response_handler(response, url, **kwargs):
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


def api_versium_com(response, url, **kwargs):
    raw_json = kwargs.pop('raw_json', False)
    if "Versium" in response:
        response["versium"] = response.pop("Versium")

    if raw_json:
        data_dict = response['versium']
    elif 'results' in response['versium'] and isinstance(response['versium']['results'][0], dict):
        data_dict = response['versium']['results'][0]
    else:
        data_dict = {}

    return data_dict
