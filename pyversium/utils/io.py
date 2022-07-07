import glob
import json
import logging
import os
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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


class ScoringPaths:
    """
    ScoringPaths class generates paths for scoring files and folders.

    Parameters
    ----------
    model_folder : str
        BinaryModel folder.

    scoring_name : str
        Name of the scoring that the scoring folder will be named for.
    """

    def __init__(self, model_folder, scoring_name):
        # Folders.
        scoring_name = str(scoring_name)
        self.scoring_folder = os.path.join(model_folder, scoring_name)

        # Create folders.
        os.makedirs(self.scoring_folder, exist_ok=True)

        # Input & output files.
        self.input_data_file = os.path.join(self.scoring_folder, 'input_data.txt')
        self.appended_data_file = os.path.join(self.scoring_folder, 'appended_data.parquet')
        self.scored_appended_data_file = os.path.join(self.scoring_folder, 'scored_appended_data.parquet')
        self.scores_file = os.path.join(self.scoring_folder, 'scores.txt')
        self.scored_data_file = os.path.join(self.scoring_folder, 'scored_data.txt')

        return


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
        self.__version__ = 1.0
        #self.__version__ = pkg_resources.get_distribution('analytics').version
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

    def get_scoring_paths(self, scoring_name):
        return ScoringPaths(self.model_folder, scoring_name)

    def __repr__(self):
        return f"ModelPaths ('{self.model_folder}')"

    def __hash__(self):
        return hash(self.model_folder)

    def __eq__(self, other):
        if not isinstance(other, ModelPaths):
            return NotImplemented
        return self.model_folder == other.model_folder


def get_output_file_format(output_file: str, input_file: str) -> (str, int):

    if input_file:
        # Split input file path into parts
        input_dir, input_base = os.path.split(input_file)
        input_root, input_ext = os.path.splitext(input_base)

        # inject input file name (minus extension) into output filename wherever there is a '$@' string.
        output_file = output_file.replace('$@', input_root)

    # Index to start output file numbering
    start_idx = 0

    # Regex for capturing output format strings between curly braces like my_file_{STARTIDX:FORMATSTRING}_output.txt
    # Attempts to split into 3 groups: '{', 'STARTIDX', ':FORMATSTRING}'
    format_capture_regex = re.compile(r"(\{)(\d*)(:\d*[a-z]\})")
    captured_format_patterns = format_capture_regex.findall(output_file)

    if captured_format_patterns:
        # If there are multiple format patterns, ignore all but the first
        captured_format_patterns = captured_format_patterns[0]
        # If we have 3 capture groups then the 2nd one signifies the start_index. Need to store this and remove it from the string.
        if len(captured_format_patterns) == 3:
            start_idx = int(captured_format_patterns[1])

            # remove the start_index portion of the format string by replacing the entire thing with just capture groups 1 and 3
            output_file = format_capture_regex.sub(r'\1\3', output_file)

    return output_file, start_idx


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

    def __init__(self, filename_or_pattern: Any, input_file=None):

        self.format_output = False
        output_file = filename_or_pattern
        start_idx = 0

        if isinstance(filename_or_pattern, str):
            self.format_output = True
            output_file, start_idx = get_output_file_format(filename_or_pattern, input_file)

            # Make all non-existing directories on the path to output_file
            if os.path.dirname(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

        self.start_idx = start_idx
        self.current = start_idx
        self.output_file = output_file

    def _format(self):
        if self.format_output:
            return self.output_file.format(self.current)
        else:
            return self.output_file

    def __iter__(self):
        self.current = self.start_idx
        return self

    def __next__(self):
        output_file = self._format()
        self.current += 1
        return output_file

    def __str__(self):
        output = self._format()
        return output.__repr__()

    @staticmethod
    def help():
        instruction = "Output file may take an absolute or relative path with optional formatting strings.\n" \
                     "If an input_file is provided, you can use $@ to have the input filename injected into the output file name\n" \
                     "A format string of the form {<STARTIDX>:FORMATSTRING} can be included in the output filename to control\n" \
                     "file numbering. FORMATSTRING is the 'new style' python string formatting.\n" \
                     "Example:\n Make a series of output_files starting at 3 padded to 3 zeros with input_file='my_file.txt'\n" \
                     "$@_output_{3:03d}.txt"
        return instruction





