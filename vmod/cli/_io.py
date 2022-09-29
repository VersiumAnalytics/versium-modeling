"""_io.py: IO helper functions for CLI tools."""
# TODO check for StopIteration when chunksize * chunkstart is greater than file size
import io
import logging
from typing import Iterator, Tuple, Literal
import sys
import itertools
from ..utils.io import OutputFileGenerator, truncate_stream

logger = logging.getLogger(__name__)

# Iterator of tuples consisting of io stream or string path to file, and the write mode for that stream/file
PairedOutputModeIteratorType = Iterator[Tuple[str | io.IOBase, Literal['w', 'a']]]


def get_output_generator(output: str | io.IOBase | None,
                         input_file: str = "",
                         chunksize: int | None = None,
                         chunkstart: int = 0,
                         truncate: bool = False,
                         delimiter: str = "\t",
                         has_header: bool = True) -> PairedOutputModeIteratorType:
    """Helper function to assist with outputting chunks of data either through Python IO objects or Pandas.to_csv method. Constructs an
    iterator that returns tuples of (output, filemode) where `output` is a stream or string path to a file, and `filemode` is either 'w' for
    write or 'a' for append. Files should be opened with mode=`filemode`. In 'write' mode, the header should always be written out, while in
    'append' mode the header should be omitted.

    For long-running processes, it is useful to be able to restart from a specific point should a job fail partway through. To avoid
    overwriting the parts of the job that already succeeded, the output needs to begin at the appropriate point. This is controlled through
    the `chunksize` and `chunkstart` options. `chunksize` can be any non-zero positive integer, but should not be changed if attempting to
    resume from a previous failure point. `chunkstart` specifies which chunk to begin at, starting from 0. If writing to multiple files,
    this will increment file numbering accordingly so that the starting output file matches the appropriate chunk. If writing to a single file,
    output will be appended to the end of the file. If the `truncate` parameter is True, everything after the first chunksize * chunkstart
    lines (+1 for the header) will be deleted, so that chunks appended to the file will be in the appropriate position. For example, if
    `chunksize` is 10 and `chunkstart` is 2, the file will be truncated after line 21 (+1 for the header) and output will begin writing to
    line 22.



    Parameters
    ----------
    output : Stream or string path to an output file.
    input_file : Name of the input file. Used for substitution in output file names.
    chunksize : The size of a chunk in number of lines
    chunkstart : Which chunk to begin iterating from.
    truncate : Whether to truncate the file after the chunkstart point.
    delimiter : Field delimiter
    has_header : Whether the stream or file contains a header. Used in conjunction with chunkstart and chunksize to determine where
        in a file to truncate data.

    Returns
    -------
    Iterator of output, filemode tuples where `output` is a stream or string path to a file. `filemode` is the mode to open the file in if
        `output` is a filepath, otherwise, `filemode` is None.
    """
    if isinstance(output, str):
        output_gen = OutputFileGenerator(output, input_file=input_file, chunkstart=chunkstart, chunksize=chunksize)

        # TODO figure out how to truncate file when there are multi-line csv records. Could do this in pure Python, but that would be slow.
        # Right now it reads to the specified line number, calculates the number of bytes, and truncates the file. When there are multi-line
        # records, then the number of lines does not equal the number of records. This will cause us to truncate too many records in the
        # output file, or possibly even delete half of a record.
        if not output_gen.is_multi_file and chunksize and chunkstart and truncate:
            raise IOError(f"Cannot resume from chunk {chunkstart} when outputting to a single file.")
            # has_header = 1 if has_header else 0
            # truncate_stream(output_gen.current_output, chunksize * chunkstart + has_header, delimiter=delimiter)  # +1 to include header:
    else:
        if output is None:
            logger.debug("No stream provided. Setting stream to stdout.")
            output = sys.stdout
        else:
            logger.debug(f"Output stream is {repr(output)}")
        output_gen = itertools.chain(itertools.repeat((output, 'w'), 1), itertools.repeat((output, 'a')))

    return output_gen
