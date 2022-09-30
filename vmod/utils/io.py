"""io.py: IO utilities."""
import csv
import glob
import io
import json
import logging
import os
import re
from typing import Iterator, Tuple

import numpy as np
import pkg_resources

logger = logging.getLogger(__name__)


def format_output_file(output_file: str, input_file: str) -> (str, int):

    if input_file:
        # Split input file path into parts
        input_dir, input_base = os.path.split(input_file)
        input_root, input_ext = os.path.splitext(input_base)

        # inject input file name (minus extension) into output filename wherever there is a '$@' string.
        output_file = output_file.replace('$@', input_root)

    # Index to start output file numbering. If output file is not formatted for numbering, set start_idx to None
    start_idx = None

    # Regex for capturing output format strings between curly braces like my_file_{STARTIDX:FORMATSTRING}_output.txt
    # Attempts to split into 3 groups: '{', 'STARTIDX', ':FORMATSTRING}'
    format_capture_regex = re.compile(r"(\{)(\d*)(:\d*[a-z]\})")
    captured_format_patterns = format_capture_regex.findall(output_file)

    if captured_format_patterns:
        start_idx = 0
        # If there are multiple format patterns, ignore all but the first
        captured_format_patterns = captured_format_patterns[0]
        # If we have 3 capture groups then the 2nd one signifies the start_index. Need to store this and remove it from the string.
        if len(captured_format_patterns) == 3:
            start_idx = int(captured_format_patterns[1].strip() or '0')

            # remove the start_index portion of the format string by replacing the entire thing with just capture groups 1 and 3
            output_file = format_capture_regex.sub(r'\1\3', output_file)

    return output_file, start_idx


def _truncate_stream(stream: io.IOBase, num_lines: int, delimiter='\t', quotechar='"'):
    stream.seek(0)
    #csv_reader = csv.reader(stream, delimiter=delimiter, dialect="excel", quotechar=quotechar)
    logger.debug(f"delimiter is {delimiter}")
    csv_reader = csv.reader(stream, delimiter=delimiter)
    i = 0
    byte_count = 0
    while i < num_lines:
        temp = next(csv_reader)
        if temp:
            byte_count += len(delimiter.join(temp).encode('latin-1'))
        i += 1
    read_lines = csv_reader.line_num
    logger.debug(f"Number of lines read: {read_lines}")
    logger.debug(f"Byte count is: {byte_count}")
    stream.flush()
    stream.seek(0)

    logger.debug(f"Stream tell is {stream.tell()}")
    stream.truncate(byte_count)
    return
    i = 0
    while i < read_lines:
        stream.readline()
        i += 1
    logger.debug(f"Stream tell is {stream.tell()}")
    stream.flush()
    stream.truncate(byte_count)
    logger.debug(f"Truncated to {stream.tell()} bytes")
    stream.flush()


def truncate_stream(file: str | io.IOBase, num_lines: int, delimiter: str = "\t", quotechar: str = '"'):
    if isinstance(file, str):
        logger.debug(f"Truncating {file} to {num_lines} lines.")
        with open(file, 'a+', newline='') as f:
            _truncate_stream(f, num_lines, delimiter, quotechar)
    # If we have an IO stream, only truncate if seek operations are allowed and the stream is not connected to a terminal/tty
    elif isinstance(file, io.IOBase) and file.seekable() and file.writable() and not file.isatty():
        file.seek(0)
        _truncate_stream(file, num_lines)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return {"<__converter__>": "ndarray", "<__args__>": [obj.tolist()], "<__kwargs__>": {"dtype": str(obj.dtype)}}
        return json.JSONEncoder.default(self, obj)


class NumpyDecoder(json.JSONDecoder):
    """ Special json decoder for numpy arrays """
    def __init__(self, **kwargs):
        kwargs['object_hook'] = self.decode_hook
        super().__init__(**kwargs)

    @staticmethod
    def decode_hook(obj):
        if obj.get("<__converter__>", "") == "ndarray":
            return np.ndarray(*obj["<__args__>"], **obj["<__kwargs__>"])
        else:
            return obj


class ModelPaths:
    """
    ModelPaths class generates paths for _model files and folders.

    Parameters
    ----------
    root_path : str
        Root path of where the _model folder will be created.

    model_name : str
        Name of the _model that the _model folder will be named for.
    """

    def __init__(self, root_path, logging_path=None):
        # Folders.
        self.root = os.path.abspath(root_path)
        self.model_folder = os.path.join(self.root, 'model')
        self.report_folder = os.path.join(self.root, 'report')

        if logging_path is None:
            self.logging_folder = os.path.join(self.root, 'logs')
        else:
            self.logging_folder = logging_path

        self.model_log_file = os.path.join(self.logging_folder, 'model.log')
        self.train_log_file = os.path.join(self.logging_folder, 'train.log')
        self.score_log_file = os.path.join(self.logging_folder, 'score.log')

        # Metadata
        try:
            self.__version__ = pkg_resources.get_distribution('vmod').version
        except pkg_resources.DistributionNotFound:
            self.__version__ = None

        self.metadata = os.path.join(self.model_folder, 'metadata.json')

        self.model_file = os.path.join(self.model_folder, 'model.pkl')
        self.model_factory = os.path.join(self.model_folder, 'model_factory.pkl')
        self.estimator_factory = os.path.join(self.model_folder, 'estimator_factory.pkl')

        # Create folders
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.report_folder, exist_ok=True)
        os.makedirs(self.logging_folder, exist_ok=True)

        if not os.path.exists(self.metadata):
            with open(self.metadata, 'w') as f:
                json.dump({'version': self.__version__}, f)
        return

    def set_metadata(self, meta_dict):
        with open(self.metadata, 'r+') as file:
            try:
                old_meta = json.load(file)
                logger.debug(f'Changing metadata. Previous metadata was:\n{old_meta}')
            except json.JSONDecodeError:
                logger.debug('No previous metadata to overwrite.')

            file.truncate(0)
            json.dump(meta_dict, file)
            logger.debug(f'Changed metadata. New metadata is:\n{meta_dict}')

    def get_metadata(self):
        with open(self.metadata, 'r') as f:
            return json.load(f)

    def __repr__(self):
        return f"ModelPaths ('{self.model_folder}')"

    def __hash__(self):
        return hash(self.model_folder)

    def __eq__(self, other):
        if not isinstance(other, ModelPaths):
            return NotImplemented
        return self.model_folder == other.model_folder


class InputFileGenerator:
    def __init__(self, filename_or_pattern: str):
        matches = glob.glob(filename_or_pattern)
        if not matches:
            raise IOError(f"Could not find any files matching {filename_or_pattern}.")
        self.inputs = matches
        self.current = 0

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.inputs):
            raise StopIteration
        result = self.inputs[self.current]
        self.current += 1
        return result


class OutputFileGenerator:
    """Takes an absolute or relative path with optional formatting strings and generates output file(s).
    If an input file is provided, you can use $@ to have the input filename injected into the output file name
    A format string of the form {<STARTIDX>:FORMATSTRING} can be included in the output filename to control
    file numbering. FORMATSTRING is the 'new style' python string formatting. For example, to make a series of output_files starting at 3
    padded to 3 zeros with input_file='my_file.txt': $@_output_{3:03d}.txt

    Multiple files will be generated if a format string is present in the filename and `chunksize` is non-zero. Otherwise, a single file is
    generated.
    """
    _multi_file: bool
    _format_output: bool
    chunkstart: int
    chunksize: int | None
    input_file: None | str
    start_idx: int
    current: int
    _output_file: str

    def __init__(self, filename_or_pattern: str,
                 chunkstart: int = 0,
                 chunksize: int | None = None,
                 input_file: str | None = None):
        if chunkstart and not chunksize:
            # Can't have a chunkstart if there is no chunksize
            raise ValueError(f"`chunkstart` was set to {chunkstart} but `chunksize` is {chunksize}. Please remove `chunkstart` parameter or"
                             " set a non-zero `chunksize`")

        self._format_output = False
        output_file = filename_or_pattern
        start_idx = None

        if isinstance(filename_or_pattern, str):
            self._format_output = True
            output_file, start_idx = format_output_file(filename_or_pattern, input_file)

            # Make all non-existing directories on the path to _output_file
            if os.path.dirname(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

        self._multi_file = start_idx is not None
        self.chunkstart = chunkstart
        self.chunksize = chunksize

        self.start_idx = start_idx or 0
        self.start_idx += chunkstart

        self.current = self.start_idx
        self._output_file = output_file

        if self.is_multi_file:
            logger.debug(f"{self.__class__.__name__} will generate multiple files starting with: {self.current_output}")
        else:
            logger.debug(f"{self.__class__.__name__} will repeatedly generate one file: {self.current_output}")

    @property
    def is_multi_file(self) -> bool:
        return self._multi_file

    @property
    def current_output(self) -> str:
        return self.format(self.current)

    def format(self, num: int):
        if self._format_output:
            return self._output_file.format(num)
        else:
            return self._output_file

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        self.current = self.start_idx
        return self

    def __next__(self) -> Tuple[str, str]:
        output_file = self.format(self.current)
        mode = 'w'

        if self.current and not self.is_multi_file:
            mode = 'a'

        self.current += 1
        return output_file, mode

    def __str__(self) -> str:
        return self.current_output
