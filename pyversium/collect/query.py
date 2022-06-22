"""qclient.py: Includes a class to query APIs asynchronously."""
import asyncio
import logging
import sys
import time
import urllib

import numpy as np
import pandas as pd
from aiohttp import ClientSession

from . import response_handlers as rhs
from .response_handler_map import RESPONSE_HANDLER_MAP

logger = logging.getLogger(__name__)


class RateLimiter(object):
    def __init__(self, calls=5, period=1, n_connections=1):
        self.calls = calls
        self.period = period
        self.clock = time.monotonic
        self.last_reset = 0
        self.num_calls = 0
        self.num_running = 0
        self.sem = asyncio.Semaphore(min(n_connections, calls))

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Semaphore will block more than {self.calls} calls from happening at once.
            async with self.sem:
                if self.num_calls + self.num_running >= self.calls:
                    await asyncio.sleep(self.__period_remaining())

                try:
                    # self.num_running is effectively a num_call that doesn't expire with the end of the period. This is to prevent us from
                    # resettting the num_calls when we have active ongoing connections.
                    self.num_running += 1
                    # func may make multiple requests, but will do so at most once per second
                    result = await func(*args, **kwargs)

                finally:
                    self.num_running -= 1
                    # Getting the remaining time and incrementing num_calls after the request causes us to overcount the amount of time since
                    # the request instead of undercount. Undercounting can lead to the server rejecting due to rate limits.
                    period_remaining = self.__period_remaining()

                    if period_remaining <= 0:
                        self.num_calls = 0
                        self.last_reset = self.clock()
                    self.num_calls += 1

                return result

        return wrapper

    def __period_remaining(self):
        elapsed = self.clock() - self.last_reset
        return self.period - elapsed


async def _fetch(session, row, query_params, url, n_retry, wait_time, read_timeout, n_connections, response_handler,
                 **kwargs):
    """Internal fetch method."""

    if query_params is None:
        query_params = {}
    row_dict = row._asdict()
    index = row_dict.pop('Index', )
    if response_handler is None or not callable(response_handler):
        response_handler = rhs.default_response_handler

    # Clean row_dict.
    # row_dict = {key: value for key, value in row_dict.items() if value not in UNKNOWN_VALUES}
    row_dict = {key: value for key, value in row_dict.items() if value is not np.nan and value is not None}
    params = {**query_params, **row_dict}

    for i in range(n_retry):

        try:
            # Wait before trying to get again. Make sure we are doing this query at most once per second or we could hit rate limits.
            await asyncio.sleep(max(wait_time, 1) * i)

            async with session.get(url, params=params, timeout=read_timeout) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)
                data_dict = response_handler(data, url=url, **kwargs)
                data_dict['__INDEX__'] = index  # Trying to avoid name collisions with 'Index' just in case.
                return data_dict
        except:
            exc = sys.exc_info()[1]
            logger.error('%s, %d attempts left: %s?%s', exc.__class__.__name__, n_retry - i, url,
                         urllib.parse.urlencode(params))

    # If cannot get data, return unsuccessful gracefully.
    logger.error('No attempts left: %s?%s', url, urllib.parse.urlencode(query_params))
    return {'__INDEX__': index}


async def _create_tasks(rows, query_params, url, *, read_timeout, n_connections, n_retry, wait_time, response_handler,
                        **kwargs):
    tasks = []
    # sem = asyncio.Semaphore(n_connections)
    limit = RateLimiter(n_connections, 1, n_connections)
    limited_fetch = limit(_fetch)
    async with ClientSession(read_timeout=read_timeout) as session:
        for row in rows:
            task = asyncio.ensure_future(
                limited_fetch(session, row, query_params, url, n_retry, wait_time, read_timeout, n_connections, response_handler, **kwargs))
            # _fetch(session, row, query_params, url, n_retry, wait_time, read_timeout, n_connections, response_handler, **kwargs))
            # _bound_fetch(sem, session, row, query_params, url, n_retry, wait_time, read_timeout, n_connections, response_handler, **kwargs))
            tasks.append(task)
        responses = asyncio.gather(*tasks)
        return await responses


class QueryClient:
    """
    QueryClient queries  APIs with given configuration asynchronously.

    Parameters
    ----------
    n_retry : int
        Number of retries for queries.
        
    retry_wait_time : int
        number of seconds to wait between retries

    n_connections : int, optional (default=20)
        Number of connections to QX.

    Raises
    ------
    ValueError
        If n_jobs is a not an integer or less than 1.
    """

    def __init__(self, url, *, params=None, n_retry=3, retry_wait_time=3, read_timeout=10, n_connections=20, mode="batch",
                 return_json=False,
                 response_handler=None, required_fields=(), optional_fields=(), field_remap=None, post_append_prefix="",
                 post_append_suffix=""):
        logger.debug('An instance of QueryClient is created at %s to query with %d connections', hex(id(self)),
                     n_connections)

        if n_connections < 1 or not isinstance(n_connections, int):
            raise ValueError('n_connections should be an integer bigger than or equal to 1, given {0}'.format(
                n_connections))

        if params is None:
            params = dict()

        # Set parameters.
        self.url = url
        self.params = params
        self.n_retry = n_retry
        self.retry_wait_time = retry_wait_time
        self.read_timeout = read_timeout
        self.n_connections = n_connections
        self.return_json = return_json
        self.required_fields = required_fields
        self.optional_fields = optional_fields
        self.post_append_prefix = post_append_prefix
        self.post_append_suffix = post_append_suffix

        if field_remap is None:
            field_remap = {}
        self.field_remap = field_remap

        # Set the response handler based on the type of response_handler
        if response_handler is None:
            # If we don't have response handler, check the RESPONSE_HANDLER_MAP for matching url, fallback to default handler
            parsed_url = urllib.parse.urlparse(url)
            netloc_path = parsed_url.netloc + parsed_url.path
            self.response_handler = RESPONSE_HANDLER_MAP.get(netloc_path, rhs.default_response_handler)

        elif isinstance(response_handler, str):
            # If response_handler is string, try to retrieve the function from the response_handlers module
            try:
                self.response_handler = getattr(rhs, response_handler)
            except AttributeError:
                raise ValueError(f"Could not find response handler function {response_handler} in {rhs.__name__}. ")

        elif callable(response_handler):
            # No need to do anything if response_handler is already a callable, If not a callable, then raise TypeError
            self.response_handler = response_handler

        else:
            raise TypeError(f"`response_handler` must be string, callable, or None. Instead got {type(response_handler)}.")

    def query(self, data):
        """
        Query an api with the data asynchronously.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to be used in queries.

        url : string
            Api endpoint to query against

        query_params : dict
            API parameters to include in the query. These will be the same between calls

        response_handler : callable

        raw_json : bool
        Returns
        -------
        pandas.DataFrame
            Response to the queries.
        """
        missing_required_columns = set(self.required_fields) - set(data.columns)
        if missing_required_columns:
            raise ValueError(f"Missing required columns {missing_required_columns} after when querying {self.url}")

        # If required_fields or optional_fields are provided, then subset the data. Otherwise use all columns
        column_subset = self.required_fields + self.optional_fields
        if column_subset:
            data = data[column_subset]

        # logger.info('Query configuration parameters are %s', json.dumps(query_params, sort_keys=True))
        # Log which columns are getting renamed.
        renamed_columns = {k: v for k, v in self.field_remap.items() if k in data}
        logger.info(f'Renaming the following columns with their respective mapping: {renamed_columns}')
        data = data.rename(columns=renamed_columns)

        n_connections = min(len(data), self.n_connections)

        logger.info('Started querying %d records', len(data))
        start_time = time.time()
        # TODO allow other index names
        rows = data.itertuples(index=True)
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(_create_tasks(rows, self.params, self.url,
                                                     read_timeout=self.read_timeout,
                                                     n_retry=self.n_retry,
                                                     wait_time=self.retry_wait_time,
                                                     n_connections=n_connections,
                                                     response_handler=self.response_handler,
                                                     raw_json=self.return_json))
        try:
            responses = loop.run_until_complete(future)

        except KeyboardInterrupt:
            # Canceling pending tasks and stopping the loop.
            asyncio.gather(*asyncio.Task.all_tasks()).cancel()
            loop.stop()

            raise KeyboardInterrupt

        if self.return_json:
            return responses

        results = pd.DataFrame(responses).set_index('__INDEX__')

        # success_rate = results['__Successful'].mean()

        time_per_query = (time.time() - start_time) / len(results) * self.n_connections
        # logger.info('Finished querying with {0:4.1%} success rate and average time per query of {1:.3f} seconds'.format(
        #    success_rate, time_per_query))
        results.columns = [self.post_append_prefix + c + self.post_append_suffix for c in results.columns]
        return results.sort_index().astype(str)
