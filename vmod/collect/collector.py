"""collector.py: Collects and prepares data from files or streams."""
import csv
import io
import logging
import os
import random
from os.path import getsize
from typing import Optional, Generator, ParamSpec, Any, Tuple

import pandas as pd
from psutil import virtual_memory

from .query import QueryClient
from ..constants import STR_NA_VALUES

logger = logging.getLogger(__name__)

P = ParamSpec('P')

DataFieldNamesTuple = Tuple[pd.DataFrame | Generator[pd.DataFrame, None, None], list[str] | None]


def _decode_stream(filepath_or_buffer: str,
                   delimiter: str,
                   **kwargs: P.kwargs) -> pd.DataFrame:
    """Attempts to character decode a file using UTF-8, falling back to latin-1 if it fails.

    Parameters
    -------
    filepath_or_buffer : str
        filepath_or_buffer to read in.

    delimiter : str
        Delimiter to use.

    kwargs : kwargs
        Additional keyword arguments to pass to pandas.read_table()

    Returns
    -------
    data : pandas.DataFrame
    """
    encoding = kwargs.pop('encoding', "utf-8")
    try:
        data = pd.read_table(filepath_or_buffer, sep=delimiter, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        logger.info('%s is not encoded with UTF-8, trying latin-1', filepath_or_buffer)
        data = pd.read_table(filepath_or_buffer, sep=delimiter, encoding='latin-1', **kwargs)

    return data


def _chunked_read(filepath_or_buffer: str,
                  delimiter: str,
                  chunksize: Optional[int],
                  **kwargs: P.kwargs) -> Generator[pd.DataFrame, None, None]:
    """A generator for reading files in chunks.

    A wrapper for reading a pandas dataframe from a file in chunks. Handles cases where part of the file is unicode
    decodable and other parts are not.

    Parameters
    -------
    filepath_or_buffer : str
        filepath_or_buffer to read in.

    delimiter : str
        Delimiter to use.

    chunksize : int
        Number of rows to process at a time.

    kwargs : kwargs
        Additional keyword arguments to pass to pandas.read_table()

    Returns
    -------
    data : pandas.DataFrame generator
    """
    n = kwargs.get('skiprows', 0)
    kwargs['header'] = None
    max_rows = kwargs.get("nrows", float("inf"))
    kwargs['nrows'] = chunksize

    total_rows = 0

    while total_rows < max_rows:
        kwargs['skiprows'] = n
        # Need to adjust chunksize if we would go past max_rows by reading the next chunk
        if isinstance(max_rows, int):
            kwargs['nrows'] = min(chunksize, max_rows - total_rows)
        try:
            data = _decode_stream(filepath_or_buffer, delimiter, **kwargs)
        except ValueError:
            break

        if len(data) < 1:
            break

        yield data
        n += chunksize
        total_rows += chunksize


def get_header(filepath_or_buffer: str,
               delimiter: str,
               quotechar: str = '"',
               dialect: str = 'unix',
               encoding: str = 'utf-8-sig') -> list[str]:
    """Get the column headers of a file.

    Assumes that the file has a header row as its first row.

    Parameters
    ----------
    filepath_or_buffer : str
        Path to the file.

    delimiter : str
        Character used to separate fields of the file.

    quotechar : str
        Character used for quoting in the file.

    dialect : str
        CSV dialect defines a set of CSV formatting properties for a system, platform, or application. Determines
        default formatting parameters such as quote characters and line terminators.

    encoding : str
        Character encoding schema to use when reading the file.

    Returns
    ------
    headers : list[str]
        Headers of the given file.
    """

    def _get_header(enc):
        with open(filepath_or_buffer, newline='', encoding=enc, mode='r+') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar, dialect=dialect)
            header = next(reader)
            logger.debug(f"Header is: {header}")
            return header

    try:
        return _get_header(encoding)
    except UnicodeDecodeError:
        return _get_header('latin-1')


def read_file_or_buffer(filepath_or_buffer: str,
                        delimiter: str,
                        safe: bool = False,
                        **kwargs: P.kwargs) -> Generator[pd.DataFrame, None, None] | pd.DataFrame:
    """Reads a file into a pandas dataframe.

    Parameters
    -------
    filepath_or_buffer : str
        filepath_or_buffer to read in.

    delimiter : str
        Delimiter to use.

    safe : bool
        Whether to use a safer but slower read for files with inconsistent encoding.

    kwargs : kwargs
        Additional keyword arguments to pass to pandas.read_table()

    Returns
    -------
    data : pandas.DataFrame
    """
    if not safe:
        return pd.read_csv(filepath_or_buffer, sep=delimiter, **kwargs)

    chunksize = kwargs.pop('chunksize', None)

    if chunksize is not None:
        return _chunked_read(filepath_or_buffer, delimiter, chunksize, **kwargs)

    else:
        return _decode_stream(filepath_or_buffer, delimiter, **kwargs)


def balance_class_fillrates(data: pd.DataFrame,
                            label_column: str,
                            label_positive_value: Any,
                            balance_fields: list[str],
                            balance_diff_tol: float = 0.1) -> pd.DataFrame:
    """
    Balance fields by randomly setting values to NaN.

    Parameters
    -------
    data :
    label_column :
    label_positive_value :
    balance_fields :
    balance_diff_tol :

    Returns
    -------
    data : pandas.DataFrame
        Balanced data.
    """
    # Get random numbers from OS. That way we don't depend on a seed.
    rnd = random.SystemRandom()

    # Get the label column as the condition.
    condition = data[label_column] == label_positive_value

    balanced_columns = []

    # Go over all fields that are set for balancing.
    for balance_column_name in balance_fields:

        logger.info(f"Column {balance_column_name} is marked for balancing")

        if balance_column_name not in data.columns:
            logger.warning(f"Column {balance_column_name} is marked for balancing but it is not in existing fields")
            continue

        # Calculate the fill ratios to check fill imbalance.
        positive_fill_ratio = data[balance_column_name][condition].notna().mean()
        negative_fill_ratio = data[balance_column_name][~condition].notna().mean()

        logger.info(
            f"Column {balance_column_name:s} has a positive fill ratio of {positive_fill_ratio:.2f}"
            f" and negative fill ratio of {negative_fill_ratio:.2f}")

        if balance_diff_tol is not None and (abs(positive_fill_ratio - negative_fill_ratio) > balance_diff_tol):

            # Revert the filter depending on which label has bigger fill ratio and calculate number of removals.
            cond_filter = condition if positive_fill_ratio > negative_fill_ratio else ~condition
            n_removal = int(abs(positive_fill_ratio - negative_fill_ratio) * cond_filter.sum())

            # Get the indices from the column to balance for random selection.
            indices = data[cond_filter][data[balance_column_name][cond_filter].notnull()].index.tolist()
            selected_indices = rnd.sample(indices, k=n_removal)

            # Set randomly selected indices to None.
            data.loc[selected_indices, balance_column_name] = None

            balanced_columns.append(balance_column_name)
            logger.info(f'Balancing column {balance_column_name}, dropped {n_removal:d} values randomly')

        else:
            logger.info(f'Not balancing column {balance_column_name}')

    return data


class Collector:
    """
    High level interface for retrieving data. Reads data from files or streams into Pandas Dataframes and performs minimal processing
    and sanity checks. Optionally performs a series of appends before returning the data by querying a set of APIs
    configured through QueryClient objects.

    Parameters
    ----------
    missing_values : list
        Data values that should be treated as missing data

    required_fields : list
        Fields that are required to be present when reading the data. Used only for sanity checking the file inputs.

    balance_fields : list
        Fields that should have their fill rates balanced across the classes in the label column. This helps to prevent
        data leakage in situations where one class has better fill rates than the other due to data collection methods and
        not true signal (e.g. converter/non-converter data where the converter data is more complete and accurate.)

    balance_diff_tol : float
        Minimum difference in fill rates between classes before balancing will be performed.

    api_queries : list of configured QueryClients
        Series of api calls to make in the given order and append to the input data.

    label : str
        Name of the label/class column

    label_positive_value : Any
        Value of the positive class

    consolidate_missing : bool
        Take all values in the data that should be treated as missing (given by `missing_values` parameter)
        and convert them to np.nan
        """

    missing_values: Optional[list[Any]]
    required_fields: list[str]
    balance_fields: list[str]
    balance_diff_tol: float
    api_queries: Optional[list[QueryClient]]
    label: Optional[str]
    label_positive_value: Optional[str]
    consolidate_missing: bool

    def __init__(self, *,
                 missing_values: Optional[list[Any]] = None,
                 required_fields: list[str] = (),
                 balance_fields: list[str] = (),
                 balance_diff_tol: float = 0.1,
                 api_queries: Optional[list[QueryClient]] = (),
                 label: Optional[str] = None,
                 label_positive_value: Optional[int | str] = None,
                 correct_label_field=True,
                 consolidate_missing=True):

        # Pre-process attributes.
        self.missing_values = STR_NA_VALUES if missing_values is None else missing_values
        self.required_fields = required_fields
        self.balance_fields = balance_fields
        self.balance_diff_tol = balance_diff_tol
        self.api_queries = api_queries
        self.label = label
        self.label_positive_value = label_positive_value
        self.correct_label_field = correct_label_field
        self.consolidate_missing = consolidate_missing

    def _check_required_fields(self, data):
        missing_fields = set(self.required_fields) - set(data.columns.values)
        if missing_fields:
            raise RuntimeError(f"Data is missing required fields: {list(missing_fields)}.")

    def _booleanize_label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Correct binary labels from their own values to 0 and 1.

        Parameters
        -------
        data : pandas.DataFrame
            Data to be corrected.

        Returns
        -------
        data : pandas.DataFrame
            Corrected data.

        Raises
        ------
        ValueError
            If there is only one type of class in the DataFrame.
        """

        # If label is already corrected, return df without any processing.
        if data[self.label].dtype != bool:
            corrected_label_column = (data[self.label] == self.label_positive_value).astype(bool)
            data[self.label] = corrected_label_column

            logger.info(
                f'Label column {self.label} positive value is corrected from {self.label_positive_value} to True.'
            )

        return data

    def append(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Append data.

        Parameters
        -------
        data : pd.DataFrame
            Input data for performing appends.

        Returns
        -------
        data : pandas.DataFrame
            Appended data.
        """
        if not self.api_queries:
            return data

        append_data = pd.DataFrame(index=data.index)
        for qclient in self.api_queries:
            api_data = qclient.query(data)
            append_data = append_data.merge(api_data, how='left', left_index=True, right_index=True, suffixes=('', '_append'))
        return data.merge(append_data, how='left', left_index=True, right_index=True, suffixes=('', '_append'))

    def collect(self, filepath_buffer_or_dataframe: str | io.TextIOBase | list[str | io.TextIOBase] | pd.DataFrame,
                has_label: bool = False,
                delimiter: str = '\t',
                header: Optional[list | tuple | str] = None,
                chunksize: Optional[int] = None,
                chunkstart: Optional[int] = None,
                safe: bool = False) -> DataFieldNamesTuple:
        """
        Read a delimited file into a Pandas DataFrame.

        Parameters
        ----------


        filepath_buffer_or_dataframe : Path to the file.

        has_label : Whether the data we are reading has a label. Should be set to True when reading training data and False otherwise

        delimiter : Delimiter.

        header : If string, this is the path to a file containing the header names. If list or tuple, this is an array of header names.
            Defaults to first line in file if None

        chunksize : Size of chunking to perform on file. If 0 or None, the entire file is read. Otherwise, the file is read in chunks and
            a dataframe generator is returned.

        chunkstart : Chunk to begin reading from, indexed from 0. If greater than 0, skips chunkstart * chunksize lines before reading file.

        safe : Tries to safely decode the data, falling back to ascii decoding if Unicode fails. This also works with chunking
            when the first half of the file properly Unicode encoded but the later half is not. When chunking, this may incur
            some additional overhead since chunking will be done via reopening the file and performing a seek and read.

        Returns
        -------
        pandas.DataFrame or pandas.DataFrame generator, names of fields in data
        """

        # If we have a dataframe, apply functions directly, otherwise it's an iterator and we need to apply a map
        def map_if_iter(function, input_data, *args, **kwargs):
            if isinstance(input_data, pd.DataFrame):
                return function(input_data, *args, **kwargs)
            else:
                return map(function, input_data, *args, **kwargs)

        logger.debug(f"filepath_buffer_or_dataframe is of type {type(filepath_buffer_or_dataframe)}")
        if isinstance(filepath_buffer_or_dataframe, pd.DataFrame):
            data = filepath_buffer_or_dataframe
            field_names = data.columns
        else:
            data, field_names = self.read(filepath_buffer_or_dataframe, delimiter=delimiter, chunksize=chunksize, chunkstart=chunkstart,
                                          header=header, safe=safe, na_values=self.missing_values)

        map_if_iter(self._check_required_fields, data)

        if has_label:
            if not self.label:
                raise ValueError("`has_label` argument set to True, but no `label` had been set for this instance of"
                                 f" {self.__class__.__name__}")

            # We can only balance fields if we are not reading in chunks.
            if not isinstance(data, pd.DataFrame) and self.balance_fields:
                raise RuntimeError("Cannot balance fields when reading in chunks. The following fields were set for balancing: "
                                   f"{self.balance_fields}")
            elif self.balance_fields:  # Only occurs if data is a Dataframe and there are fields selected for balancing
                for b in self.balance_fields:
                    if b not in data:
                        raise ValueError(f'Column {b} was selected for balancing but was not found in the data.')

                data = balance_class_fillrates(data, self.label, self.label_positive_value, self.balance_fields, balance_diff_tol=0.1)

            if self.correct_label_field:
                data = map_if_iter(self._booleanize_label, data)

        data = map_if_iter(self.append, data)

        return data, field_names

    @staticmethod
    def read(filepath_or_buffer: str | io.TextIOBase | list[str | io.TextIOBase],
             delimiter: str = '\t',
             header: Optional[list | tuple | str] = None,
             chunksize: Optional[int] = None,
             chunkstart: Optional[int] = None,
             safe: bool = False,
             na_values: list[str] = ()) -> DataFieldNamesTuple:
        """
        Read a delimited file into a Pandas DataFrame.

        Parameters
        ----------

        filepath_or_buffer : Path to a file or a buffer. Alternatively, a list of file paths or buffers.

        delimiter : Delimiter.

        header : If string, this is the path to a file containing the header names. If list or tuple, this is an array of header names.
            Defaults to first line in file if None

        chunksize : Size of chunking to perform on file. If 0 or None, the entire file is read. Otherwise, the file is read in chunks and
            a dataframe generator is returned.

        chunkstart : Chunk to begin reading from, indexed from 0. If greater than 0, skips chunkstart * chunksize lines before reading file.

        safe : Tries to safely decode the data, falling back to ascii decoding if Unicode fails. This also works with chunking
            when the first half of the file properly Unicode encoded but the later half is not. When chunking, this may incur
            some additional overhead since chunking will be done via reopening the file and performing a seek and read.

        na_values : Values to consider missing.

        Returns
        -------
        pandas.DataFrame or pandas.DataFrame generator, names of fields in data
        """

        # If not given a header assume it's the first row in the file.
        if header is None:
            logger.debug("There is no header provided. Set header=0 and names=None")
            header = 0
            names = None
        # If header is a list or tuple, use those as column names and assume there is no header in the file
        elif isinstance(header, (list, tuple)):
            logger.debug("header is a list of column names. Setting names=header and header=None")
            names = header
            header = None
        # If header is a string, assume it's a path to a header file and attempt to read.
        elif isinstance(header, str):
            logger.debug("header is a string. Treating as a filepath and attempting to load.")
            if not os.path.exists(header):
                raise IOError(f"Could not find header file located at {header}. File does not exist.")
            names = get_header(header, delimiter=delimiter)
            header = None
        else:
            raise TypeError(f"Unrecognized type for header: {type(header)}.")

        # These will be the fields or columns in the dataset
        field_names = names

        if isinstance(filepath_or_buffer, (list, tuple)):
            logger.debug(f"filepath_or_buffer is iterable. Attempting to load {len(filepath_or_buffer)} files.")
            if chunksize:
                raise ValueError("Cannot set both multiple files and chunksize. You can either read multiple files or one file in chunks.")

        elif isinstance(filepath_or_buffer, str):
            logger.info(f'Loading dataset from file {filepath_or_buffer}')
            if field_names is None:
                # Get the field names from the file header
                field_names = get_header(filepath_or_buffer, delimiter=delimiter)
            filepath_or_buffer = [filepath_or_buffer]  # Wrap in list for consistency with multiple input files

        elif isinstance(filepath_or_buffer, io.TextIOBase):
            if field_names is None and chunksize:
                logger.warning("Could not determine field names ahead of time since dataframes are lazy-loaded chunks from a"
                               " buffered stream.")
            filepath_or_buffer = [filepath_or_buffer]  # Wrap in list for consistency with multiple input files

        else:
            raise TypeError(f"Unrecognized type for filepath_or_buffer: {type(filepath_or_buffer)}.")

        dataframes = []
        skiprows = 0
        if chunksize and chunkstart:
            skiprows = chunkstart * chunksize

        # Check if we have enough memory.
        max_file_size = max(map(getsize, filepath_or_buffer))
        low_memory = (max_file_size > virtual_memory().available * 0.75)
        logger.debug(f"Low memory is {low_memory}")

        for f in filepath_or_buffer:
            try:
                data = read_file_or_buffer(f, delimiter, engine='c', dtype=str,
                                           parse_dates=False, skip_blank_lines=False, low_memory=low_memory,
                                           chunksize=chunksize, header=header, names=field_names, safe=safe, na_values=na_values,
                                           skiprows=skiprows)
                if chunksize:
                    logger.debug(f"`chunksize` is {chunksize}. Returning iterable of lazy-loaded dataframes.")
                    return data, field_names
                dataframes += [data]

            except Exception as e:
                raise IOError(f'Cannot read {f}').with_traceback(e.__traceback__)

        logging.debug(f"Concatenating {len(dataframes)} dataframes together.")
        data = pd.concat(dataframes, ignore_index=True, copy=False, axis=0)

        logger.info(f"Finished loading dataset.")
        logger.info(f'Dataset contains {len(data)} records')
        field_names = list(data.columns.values)
        logger.info(f"Dataset contains {len(field_names)} fields.")
        return data, field_names
