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
from .response_handler_map import handler_from_url

logger = logging.getLogger(__name__)

MAX_QUERIES_PER_SECOND = 100
SUCCESS_IDENTIFIER = "$@&__SUCCESS__&@$"
INDEX_IDENTIFIER = "$@&__INDEX__&@$"


class RateLimiter(object):
    def __init__(self, *, max_calls=20, period=1, n_connections=100, n_retry=3, retry_wait_time=2):
        self.max_calls = max_calls
        self.period = period
        self.n_retry = n_retry
        self.retry_wait_time = retry_wait_time
        self.clock = time.monotonic
        self.last_reset = 0
        self.num_calls = 0
        self.sem = asyncio.Semaphore(n_connections)

    def __call__(self, func):

        async def wrapper(*args, **kwargs):
            # Semaphore will block more than {self.n_connections} from happening at once.
            self.last_reset = self.clock()
            async with self.sem:
                # async for allows us to move on to next record instead of looping over the same record consecutively
                #async for i in async_range(self.n_retry + 1):
                for i in range(self.n_retry + 1):
                    while self.num_calls >= self.max_calls:
                        await asyncio.sleep(self.__period_remaining())

                    try:
                        self.num_calls += 1
                        result = await func(*args, attempts_left=self.n_retry - i, **kwargs)
                        return result

                    # Doesn't matter what the exception is, we will still try again
                    except:
                        if self.n_retry - i > 0:
                            await asyncio.sleep(self.retry_wait_time * i)
                # Failed to perform the query after n_retry attempts. Return empty response
                return {SUCCESS_IDENTIFIER: False}

        return wrapper

    def __period_remaining(self):
        elapsed = self.clock() - self.last_reset
        period_remaining = self.period - elapsed
        if period_remaining <= 0:
            self.num_calls = 0
            self.last_reset = self.clock()
            period_remaining = self.period
        return max(period_remaining, 0)


async def _fetch(session, row, query_params, url, read_timeout, response_handler, headers, attempts_left,
                 **kwargs):
    """Internal fetch method."""
    if query_params is None:
        query_params = {}
    row_dict = row._asdict()
    idx = row_dict.pop(INDEX_IDENTIFIER, None)

    if response_handler is None or not callable(response_handler):
        response_handler = rhs.default_response_handler

    # Clean row_dict.
    # row_dict = {key: value for key, value in row_dict.items() if value not in UNKNOWN_VALUES}
    row_dict = {key: value for key, value in row_dict.items() if value is not np.nan and value is not None}
    params = {**query_params, **row_dict}

    try:
        async with session.get(url, params=params, timeout=read_timeout, headers=headers) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)

            if data is None:
                raise ValueError("Nothing returned in response")

            data_dict = response_handler(data, url=url, **kwargs)
            data_dict[SUCCESS_IDENTIFIER] = True
            return data_dict

    except Exception as e:
        logger.error(f"Error during url fetch: {e}\nIndex: {idx}\nURL: {url}?{urllib.parse.urlencode(params)}"
                     f"\nResponse Status: {response.status}\nAttempts Left: {attempts_left:d}")
        raise


async def _create_tasks(rows, query_params, url, *, read_timeout, n_connections, n_retry, retry_wait_time, response_handler, headers=None,
                        queries_per_second=20, **kwargs):
    tasks = []
    limit = RateLimiter(max_calls=queries_per_second, period=1, n_connections=n_connections, n_retry=n_retry,
                        retry_wait_time=retry_wait_time)
    limited_fetch = limit(_fetch)
    
    async with ClientSession(read_timeout=read_timeout) as session:
        for row in rows:
            task = asyncio.ensure_future(
                limited_fetch(session=session, row=row, query_params=query_params, url=url, read_timeout=read_timeout,
                              response_handler=response_handler, headers=headers, **kwargs))

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

    def __init__(self, url, *, params=None, headers=None, n_retry=3, retry_wait_time=3, read_timeout=10, n_connections=100,
                 queries_per_second=20, return_json=False, response_handler=None, required_fields=(), optional_fields=(), field_remap=None,
                 post_append_prefix="", post_append_suffix=""):

        if n_connections < 1 or not isinstance(n_connections, int):
            raise ValueError(f"n_connections should be an integer greater than or equal to 1, given {n_connections}")

        if params is None:
            params = dict()

        # Set parameters.
        self.url = url
        self.params = params
        self.n_retry = n_retry
        self.headers = headers
        self.retry_wait_time = retry_wait_time
        self.read_timeout = read_timeout
        self.n_connections = n_connections
        self.return_json = return_json
        self.required_fields = required_fields
        self.optional_fields = optional_fields
        self.post_append_prefix = post_append_prefix
        self.post_append_suffix = post_append_suffix
        
        if queries_per_second > MAX_QUERIES_PER_SECOND:
            raise ValueError(f"`queries_per_second` set to {queries_per_second}. Maximum allowed: {MAX_QUERIES_PER_SECOND}")
        self.queries_per_second = queries_per_second

        if field_remap is None:
            field_remap = {}
        self.field_remap = field_remap

        # Set the response handler based on the type of response_handler
        if response_handler is None:
            self.response_handler = handler_from_url(url)

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

        # Log which columns are getting renamed.
        renamed_columns = {k: v for k, v in self.field_remap.items() if k in data}
        logger.info(f'Renaming the following columns with their respective mapping: {renamed_columns}')
        data = data.rename(columns=renamed_columns)

        # Change index name so it does not
        index_name = data.index.name
        data.index.name = INDEX_IDENTIFIER
        n_connections = min(len(data), self.n_connections)

        logger.info(f"Started querying {self.url} with {len(data)} records.")
        logger.debug(f"HTTP headers are set to {self.headers}")
        start_time = time.monotonic()

        rows = data.itertuples(index=True)
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(_create_tasks(rows, self.params, self.url,
                                                     headers=self.headers,
                                                     read_timeout=self.read_timeout,
                                                     n_retry=self.n_retry,
                                                     retry_wait_time=self.retry_wait_time,
                                                     n_connections=n_connections,
                                                     response_handler=self.response_handler,
                                                     queries_per_second=self.queries_per_second,
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

        results = pd.DataFrame(responses)
        results.index.name = index_name # Set index name back to
        end_time = time.monotonic()

        success_rate = results[SUCCESS_IDENTIFIER].mean()
        results.drop(columns=[SUCCESS_IDENTIFIER], inplace=True)
        
        query_time = (end_time - start_time)
        num_results = len(results)
        actual_queries_per_second = num_results / query_time

        logger.info(f"Finished querying {num_results} records in {query_time:.1f} seconds with success rate {success_rate:.2%}."
                    f" Averaged {actual_queries_per_second:.2f} queries per second.")
        # logger.info('Finished querying with {0:4.1%} success rate and average time per query of {1:.3f} seconds'.format(
        #    success_rate, time_per_query))
        results.columns = [self.post_append_prefix + c + self.post_append_suffix for c in results.columns]
        return results.sort_index().astype(str)
