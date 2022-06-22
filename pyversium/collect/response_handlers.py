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

def api2b_versium_com_q2(response, url, **kwargs):
    raw_json = kwargs.pop('raw_json', False)

    if raw_json:
        data_dict = response['Versium']
    elif 'results' in response['Versium'] and isinstance(response['Versium']['results'][0], dict):
        data_dict = response['Versium']['results'][0]
    else:
        data_dict = {}


    query_time = response['Versium']['query-time']
    logger.debug('Successful query in %s seconds: %s', query_time, url)

    return data_dict
