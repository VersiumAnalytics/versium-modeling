"""qclient.py: Includes a class to query APIs asynchronously."""
import asyncio
import logging
import time
import urllib
from typing import Optional, Callable, NamedTuple, Any

import aiohttp
import numpy as np
import pandas as pd
from aiohttp import ClientSession, ClientResponseError

from . import response_handlers as rhs
from .response_handlers import ResponseHandlerType
from .response_handler_map import handler_from_url

logger = logging.getLogger(__name__)

MAX_QUERIES_PER_SECOND = 100

QueryResult = NamedTuple("QueryResult", [('result', dict | list), ('success', bool)])
QueryInput = NamedTuple("QueryInput", [('Index', int | float), ('data', Any)])


class QueryError(RuntimeError):
    pass


class RateLimiter(object):

    def __init__(self,
                 *,
                 max_calls: int = 20,
                 period: int | float = 1,
                 n_connections: int = 100,
                 n_retry: int = 3,
                 retry_wait_time: int | float = 2):

        self.max_calls = max_calls
        self.period = period
        self.n_retry = n_retry
        self.retry_wait_time = retry_wait_time
        self.clock = time.monotonic
        self.last_reset = 0
        self.num_calls = 0
        self.sem = asyncio.Semaphore(n_connections)

    def __call__(self, func: Callable):

        async def wrapper(*args, **kwargs):
            # Semaphore will block more than {self.n_connections} from happening at once.
            self.last_reset = self.clock()
            async with self.sem:
                for i in range(self.n_retry + 1):
                    while self.num_calls >= self.max_calls:
                        await asyncio.sleep(self.__period_remaining())

                    try:
                        self.num_calls += 1
                        result = await func(*args, attempts_left=self.n_retry - i, **kwargs)
                        return result

                    except QueryError:
                        if self.n_retry - i > 0:
                            await asyncio.sleep(self.retry_wait_time * i)

                # Failed to perform the query after n_retry attempts. Return empty response
                return QueryResult(result={}, success=False)

        return wrapper

    def __period_remaining(self):
        elapsed = self.clock() - self.last_reset
        period_remaining = self.period - elapsed
        if period_remaining <= 0:
            self.num_calls = 0
            self.last_reset = self.clock()
            period_remaining = self.period
        return max(period_remaining, 0)


async def _fetch(session: aiohttp.ClientSession,
                 record: QueryInput,
                 query_params: dict,
                 url: str,
                 response_handler: ResponseHandlerType,
                 headers: dict,
                 attempts_left: int,
                 required_fields: Optional[set[str]] = None,
                 **kwargs):
    """Internal fetch method."""
    if query_params is None:
        query_params = {}
    idx = record.Index
    rec = {key: value for key, value in record.data.items() if value is not np.nan and value is not None}

    if response_handler is None or not callable(response_handler):
        response_handler = rhs.default_response_handler

    # Check if missing required fields
    if required_fields is not None:
        missing_required_fields = required_fields - set(rec)
        if missing_required_fields:
            logger.debug(f"Missing required fields: {missing_required_fields}\n\tIndex: {idx}\n\tRecord: {rec}")
            return QueryResult(result={}, success=False)

    params = {**query_params, **rec}
    response = None

    try:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)
            if data is None:
                raise QueryError("Nothing returned in response")

            result = response_handler(data, url=url, **kwargs)
            return QueryResult(result=result, success=True)

    except Exception as e:
        if isinstance(e, ClientResponseError):  # Log some extra info if it's a ClientResponseError
            logger.error(f"Error during url fetch: {e.message}\n\tURL: {e.request_info.real_url}"
                         f"\n\tResponse Status: {e.status}\n\tHeaders: {e.headers}\n\tAttempts Left: {attempts_left:d}")
        else:
            status = getattr(response, "status", "UNKNOWN")
            logger.error(f"Error during url fetch: {e}\n\tURL: {url}?{urllib.parse.urlencode(params)}"
                         f"\n\tResponse Status: {status}\n\tAttempts Left: {attempts_left:d}")
        # Use a different error class so that we only catch errors from making the http request
        raise QueryError("Failed to fetch url.")


async def _create_tasks(records: list[QueryInput],
                        query_params: dict,
                        url: str,
                        *,
                        timeout: int | float,
                        n_connections: int,
                        n_retry: int,
                        retry_wait_time: float | int,
                        response_handler: ResponseHandlerType,
                        headers: Optional[dict] = None,
                        queries_per_second: float | int = 20,
                        **kwargs):
    tasks = []
    limit = RateLimiter(max_calls=queries_per_second, period=1, n_connections=n_connections, n_retry=n_retry,
                        retry_wait_time=retry_wait_time)
    limited_fetch = limit(_fetch)
    
    async with ClientSession(read_timeout=timeout) as session:
        for rec in records:
            task = asyncio.ensure_future(
                limited_fetch(session=session, record=rec, query_params=query_params, url=url,
                              response_handler=response_handler, headers=headers, **kwargs))

            tasks.append(task)
        responses = asyncio.gather(*tasks)
        return await responses


class QueryClient:
    """
    Queries APIs asynchronously.

    Parameters
    ----------
    url : API endpoint

    params : API parameters to pass with every record

    headers : HTTP headers to include with every record

    n_retry : Number of times to retry an unsuccessful query

    retry_wait_time : Number of seconds to wait between retries. The wait time will increase by a factor of
        `retry_wait_time` everytime the call fails. For example, if `retry_wait_time`=3, then it will wait 0 seconds on the first retry, 3
        seconds on the second retry, 6 seconds on the third retry, etc.

    timeout :  Number of seconds to wait for a response before timing out.

    n_connections : Maximum number of total active function calls to allow. Once this limit is reached, no more function calls will be made
        until an active call returns.

    queries_per_second : Number of queries to perform each second

    return_json : Return the responses as raw JSON instead of unpacking into a Dataframe

    response_handler : Callable or name of function defined in `response_handlers` module. Will be called on every API response for processing

    field_remap : Mapping of input field names to API parameter names.

    required_fields : Fields that are required to be present to query the API. If any of these are missing, that record will not be queried
        but will still be included in the output with appended fields empty.
    optional_fields : Fields that will be included in the API query if they are present

    post_append_prefix : String to prefix appended field names with. Used to differentiate duplicate field names and denote where a field came
        from when performing multiple appends.

    post_append_suffix : String to suffix appended fields names with. Used to differentiate duplicate field names and denote where a field came
        from when performing multiple appends.
    """
    url: str
    params: Optional[dict]
    headers: Optional[list[str]]
    n_retry: int
    retry_wait_time: int
    timeout: float | int
    n_connections: int
    queries_per_second: float | int
    return_json: bool
    response_handler: Optional[ResponseHandlerType | str]
    required_fields: Optional[list[str]]
    optional_fields: Optional[list[str]]
    field_remap: Optional[dict[str, str]]
    post_append_prefix: str
    post_append_suffix: str

    def __init__(self, url: str,
                 *,
                 params: Optional[dict] = None,
                 headers: Optional[list[str]] = None,
                 n_retry: int = 3,
                 retry_wait_time: int = 3,
                 timeout: float | int = 10.0,
                 n_connections: int = 100,
                 queries_per_second: float | int = 20,
                 return_json: bool = False,
                 response_handler: Optional[ResponseHandlerType | str] = None,
                 required_fields: Optional[list[str]] = None,
                 optional_fields: Optional[list[str]] = None,
                 field_remap: Optional[dict[str, str]] = None,
                 post_append_prefix: str = "",
                 post_append_suffix: str = ""):

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
        self.timeout = timeout
        self.n_connections = n_connections
        self.return_json = return_json
        self.required_fields = required_fields
        self.optional_fields = optional_fields
        self.post_append_prefix = post_append_prefix
        self.post_append_suffix = post_append_suffix
        
        if queries_per_second > MAX_QUERIES_PER_SECOND:
            raise ValueError(f"`queries_per_second` set to {queries_per_second}. Maximum allowed: {MAX_QUERIES_PER_SECOND}")
        self.queries_per_second = queries_per_second

        # Correct required and optional fields
        if self.required_fields is None:
            self.required_fields = []
        elif isinstance(self.required_fields, str):
            self.required_fields = [self.required_fields]

        if self.optional_fields is None:
            self.optional_fields = []
        elif isinstance(self.optional_fields, str):
            self.optional_fields = [self.optional_fields]

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

    def query(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Query an api with the data asynchronously.

        Parameters
        ----------
        data : Data to be used in queries.

        Returns
        -------
        Response to the queries.
        """
        missing_required_columns = set(self.required_fields) - set(data.columns)
        if missing_required_columns:
            raise ValueError(f"Missing required columns {missing_required_columns} when querying {self.url}")

        # If required_fields or optional_fields are provided, then subset the data. Otherwise, use all fields
        column_subset = self.required_fields + self.optional_fields
        if column_subset:
            data = data[column_subset]

        # Log which fields are getting renamed.
        renamed_columns = {k: v for k, v in self.field_remap.items() if k in data}
        logger.info(f'Renaming the following fields with their respective mapping: {renamed_columns}')
        data = data.rename(columns=renamed_columns)

        # Rename the required fields using the field_remap
        logger.info(f"Fields required to be non-empty for the api call: {self.required_fields}")
        required_fields = [self.field_remap.get(k, k) for k in self.required_fields]
        logger.info(f"After applying the field remap, the new required fields are {required_fields}")

        # Change index name to not collide with column names
        original_index = data.index.copy()

        n_connections = min(len(data), self.n_connections)

        logger.info(f"Started querying {self.url} with {len(data)} records.")
        logger.debug(f"HTTP headers are set to {self.headers}")
        start_time = time.monotonic()

        # Create list of QueryInput tuples from zipped index and dict
        records = [QueryInput(idx, dat) for idx, dat in zip(original_index.values, data.to_dict(orient="records"))]

        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(_create_tasks(records, self.params, self.url,
                                                     headers=self.headers,
                                                     timeout=self.timeout,
                                                     n_retry=self.n_retry,
                                                     retry_wait_time=self.retry_wait_time,
                                                     n_connections=n_connections,
                                                     response_handler=self.response_handler,
                                                     queries_per_second=self.queries_per_second,
                                                     raw_json=self.return_json,
                                                     required_fields=set(required_fields)))
        try:
            responses = loop.run_until_complete(future)

        except KeyboardInterrupt:
            # Canceling pending tasks and stopping the loop.
            asyncio.gather(*asyncio.Task.all_tasks()).cancel()
            loop.stop()

            raise KeyboardInterrupt

        end_time = time.monotonic()
        query_time = (end_time - start_time)

        results, successes = zip(*responses)

        num_results = len(results)
        success_rate = sum(successes) / num_results
        actual_queries_per_second = num_results / query_time

        logger.info(f"Finished querying {num_results} records in {query_time:.1f} seconds with success rate {success_rate:.2%}."
                    f" Averaged {actual_queries_per_second:.2f} queries per second.")

        if self.return_json:
            return responses
        results = pd.DataFrame(results, index=original_index)
        results.columns = [self.post_append_prefix + c + self.post_append_suffix for c in results.columns]
        return results
