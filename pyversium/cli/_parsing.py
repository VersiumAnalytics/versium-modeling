import argparse
import json
import logging
import re
from typing import Any
from typing import Optional

from ..utils import OutputFileGenerator

logger = logging.getLogger(__name__)

GLOBAL_EPILOG = """
        Multiple Arguments:\r
        Some options accept multiple arguments, such as `--required-fields` or `--missing-values`. The options will denote in their help
        documentation if they take multiple arguments. Multiple arguments should be passed to the CLI as whitespace separated arguments.
        If a single argument contains whitespace, enclose it in double or single quotes.\rFor example:\r
        \t--required-fields field_1 "field 2" field_3 'field 4'\rIn the config file, these arguments should be passed as an array.
        \r\r
        Chunking:\r
        The `--chunksize` and `--chunkstart` options are used to break up input and output into chunks of lines. This is helpful to prevent
        exploding memory for large jobs or to checkpoint a job at certain intervals in case it fails. `--chunksize` determines the size of 
        a chunk in number of lines. `chunksize` lines will be read from input, processed, and then written to output before starting on the
        next chunk. Should something fail partway through a job, it's possible to restart the job from a particular chunk with the
        `--chunkstart` option, with chunk numbering starting from 0. Passing a non-zero positive integer to `--chunkstart` will adjust 
        both input and output starting points appropriately so that results are the same as if the job were started from the very beginning.
        It is important when restarting a job this way to not change the `--chunksize` parameter, or it will mess up the output. To avoid
        this scenario, it's recommending that when chunking the `--chunksize` parameter is put into a config file instead of passed directly
        to the CLI.
"""


class WhitespaceHelpFormatter(argparse.HelpFormatter):
    """Help formatter that preserves tabs and carriage returns for more control over spacing in help strings.

    Tabs (\t) are converted to a sequence of single space characters. The number of spaces is controlled by the `tabsize` parameter. Newline
    characters (\n) are not preserved, but carriage returns (\r) can be used in place of newline characters as line separators. These
    carriage return characters will be converted to newline characters in the final output.

    This class is primarily intended for long epilogs where line breaks and indents are needed to break the epilog into different sections.
    This will also work on argument help strings, but beware that there may be indentation issues
    """

    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=24,
                 width=None,
                 tabsize=4):
        super().__init__(prog, indent_increment=indent_increment, max_help_position=max_help_position, width=width)

        # Allows tabs and carriage returns to remain.
        self._paragraph_whitespace_matcher = re.compile(r'[ \n\f\v]+', re.ASCII)
        self.tabsize = tabsize

    def _split_lines(self, text, width):
        text = self._paragraph_whitespace_matcher.sub(' ', text).strip()
        import textwrap
        result = textwrap.wrap(text, width, replace_whitespace=False, tabsize=self.tabsize)
        return [r.replace('\r', '\n') for r in result]

    def _fill_text(self, text, width, indent):
        text_new = self._paragraph_whitespace_matcher.sub(' ', text).strip()

        text_new2 = text_new.splitlines()
        import textwrap
        result = [textwrap.fill(t.strip(' \n\r'), width, initial_indent=indent, subsequent_indent=indent, tabsize=self.tabsize)
                  for t in text_new2]

        return "\n".join(result)


class ParserCollection:

    def __init__(self, main_parser, subparsers=()):
        self.main_parser = main_parser
        self.subparsers = list(subparsers)

    def parse_args(self, args):
        return self.main_parser.parse_args(args)

    def set_defaults(self, **kwargs):
        self.main_parser.set_defaults(**kwargs)
        for s in self.subparsers:
            s.set_defaults(**kwargs)


def get_config(args: list[str],
               parser: ParserCollection,
               subparser_dest: Optional[str] = None,
               all_cmd_names: Optional[list[str]] = None
               ) -> dict[str, Any]:
    parsed_args = parser.parse_args(args)

    if subparser_dest is None:
        subparser_dest = ""

    if all_cmd_names is None:
        all_cmd_names = []

    # Reparse if we have a config file, using the config file as the new defaults
    if parsed_args.config:
        with open(parsed_args.config, 'r') as f:
            config = json.load(f)

        # Get the command-specific configs for our current command if available.
        cmd_configs = {}
        cmd = getattr(parsed_args, subparser_dest, "")
        if cmd and cmd in config:
            cmd_configs = config.pop(cmd)

        # Pop subcommands from config.
        for cmd_name in all_cmd_names:
            config.pop(cmd_name, None)

        # Update the configs with the command-specific options
        config.update(cmd_configs)

        # Set the parser defaults to our config file. Anything passed to command line explicitly will overwrite these.
        parser.set_defaults(**config)

        # Reparse our CLI args using the new defaults from the config file.
        parsed_args = parser.parse_args(args)

    # Config argument is no longer needed from the parser.
    del parsed_args.config

    return vars(parsed_args)


def parser_add_output_options(parser):
    parser.add_argument('-o', '--output', action='store', type=str, default=None,
                        help="File to write output results to. Defaults to stdout. \n" + OutputFileGenerator.__doc__)


def parser_add_input_options(parser):
    parser.add_argument("-i", "--input", action="store", type=str, default=None, help="Input file.")


def parser_add_log_options(parser):
    # Logging options
    parser.add_argument("--log-level", action="store", type=str, choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        default="ERROR", help="Level for logging. (default: %(default)s)")
    parser.add_argument("--log-file", action="store", type=str, default=None, help="File to write logs to.")
    parser.add_argument("--log-packages", action="store", nargs="*", type=str, default=[],
                        help="Additional packages you want to log (e.g. numpy, pandas, sklearn). Takes multiple arguments. (default: %(default)s)")
    parser.add_argument("--overwrite-logs", action="store_true", help="Overwrite the log files instead of appending (default: %(default)s)")


def parser_add_collector_options(parser, base=True, chunk=True, label=True, balance=True):
    # Options needed by Collector constructor
    if base:
        parser.add_argument('--header', type=str, default=None,
                            help="Path to header file, if there is one. Don't include this if your file already has a header.")
        parser.add_argument('-d', '--delimiter', action='store', type=str, default=",", help="Column separator for input and output. (default: %(default)s)")
        parser.add_argument('--required-fields', nargs='*', action='store', type=str, default=None,
                            help='Columns that are required to be present when reading in the data. This helps catch errors early on before'
                                 'appending data. Takes multiple arguments.')
        parser.add_argument('--missing-values', nargs='*', action='store', type=str,
                            help='Values to consider as missing or unknown (e.g. n/a None UNKNOWN). Takes multiple arguments. (default: %(default)s)',
                            default=None)

    if chunk:
        parser.add_argument('--chunksize', type=int, default=None, metavar='CHUNKSIZE',
                            help='Read input in chunks of CHUNKSIZE lines instead of all at once. Not compatible with column balancing. (default: %(default)s)')
        parser.add_argument('--chunkstart', type=int, default=0, metavar='CHUNK START',
                            help="Which chunk to start at, with 0 being the first chunk. Useful if writing out to multiple files and you want"
                                 " to resume the job from a particular point. (default: %(default)s)")

    if label:
        parser.add_argument('-l', '--label', action='store', type=str, default=None, metavar='LABEL NAME',
                            help='Name of label column, if there is one.')
        parser.add_argument('--label-positive-value', action='store', type=str, default=None, metavar='LABEL POSITIVE VALUE',
                            help='Value for the positive class.')

    if balance:
        parser.add_argument('--balance-fields', nargs='*', action='store', type=str, default=None,
                            help='Columns to perform missing value balancing on. Columns are only balanced if --label and --train are provided.'
                                 ' Not compatible with --chunksize option. Takes multiple arguments.')

        parser.add_argument('--balance-diff-tol', action='store', type=float, metavar='(0.0-1.0)', default=None,
                            help='Tolerance for difference in fill rates between classes. If the difference in fill rate is greater than this, then'
                                 ' all fields passed to --balance-fields will have their missing values balanced. This only applies if the'
                                 ' --train option is used.')


def parser_add_include_exclude_options(parser: argparse.ArgumentParser, purpose_string: str = "be used."):
    parser.add_argument("--include-fields", action="store", type=str, nargs="*", default=[],
                        help=f"Explicit fields to include. Included fields will {purpose_string}. Takes multiple arguments.")

    parser.add_argument("--exclude-fields", action="store", type=str, nargs="*", default=[],
                        help=f"Explicit fields to exclude. Excluded fields will not {purpose_string}. Takes multiple arguments.")

    parser.add_argument("--regex-include-fields", action="store", type=str, nargs="*", default=[],
                        help=f"Use regex expressions to include fields that match the pattern. Included fields will {purpose_string}. "
                             "Takes multiple arguments.")

    parser.add_argument("--regex-exclude-fields", action="store", type=str, nargs="*", default=[],
                        help=f"Use regex expressions to exclude fields that match the pattern. Excluded fields will not {purpose_string}. "
                             "Takes multiple arguments")
