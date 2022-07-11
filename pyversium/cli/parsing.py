import json
import logging
import sys
from typing import Any
from typing import Optional

from jsonschema import Draft4Validator

logger = logging.getLogger(__name__)


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


def validate_json_schema(instance, schema):
    # Check schema validity.
    Draft4Validator.check_schema(schema)

    # Create the validator.
    validator = Draft4Validator(schema)
    validator.validate(instance)

    return instance


def get_config(args: list[str],
               parser: ParserCollection,
               config_schema: dict,
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

        validate_json_schema(config, config_schema)

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


def parser_add_log_options(parser):
    # Logging options
    parser.add_argument("--log-level", action="store", type=str, choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                        default="ERROR", help="Level for logging.")
    parser.add_argument("--log-file", action="store", type=str, default=None, help="File to write logs to.")
    parser.add_argument("--log-packages", action="store", nargs="*", type=str, default=[],
                        help="Additional packages you want to log (e.g. numpy, pandas, sklearn)")
    parser.add_argument("--overwrite-logs", action="store_true", help="Overwrite the log files instead of appending")


def parser_add_collector_options(parser, base=True, chunk=True, label=True, balance=True):
    # Options needed by Collector constructor
    if base:
        parser.add_argument('--header', type=str, default=None,
                            help="Path to header file, if there is one. Don't include this if your file already has a header.")
        parser.add_argument('-d', '--delimiter', action='store', type=str, default=",", help="Column separator for input and output.")
        parser.add_argument('--required-fields', nargs='*', action='store', type=str, default=None,
                            help='Columns that are required to be present when reading in the data. This helps catch errors early on before appending data.')
        parser.add_argument('--missing-values', nargs='*', action='store', type=str,
                            help='Values to consider as missing or unknown (e.g. n/a None UNKNOWN).', default=None)

    if chunk:
        parser.add_argument('--chunksize', type=int, default=None, metavar='CHUNKSIZE',
                            help='Read input in chunks of CHUNKSIZE lines instead of all at once. Not compatible with column balancing.')

    if label:
        parser.add_argument('-l', '--label', action='store', type=str, default=None, metavar='LABEL NAME',
                            help='Name of label column, if there is one.')
        parser.add_argument('--label-positive-value', action='store', type=str, default=None, metavar='LABEL POSITIVE VALUE',
                            help='Value for the positive class.')

    if balance:
        parser.add_argument('--balance-fields', nargs='*', action='store', type=str, default=None,
                            help='Columns to perform missing value balancing on. Columns are only balanced if --label and --train are provided.'
                                 ' Not compatible with --chunksize option.')

        parser.add_argument('--balance-diff-tol', action='store', type=float, metavar='(0.0-1.0)', default=None,
                            help='Tolerance for difference in fill rates between classes. If the difference in fill rate is greater than this, then'
                                 ' all fields passed to --balance-fields will have their missing values balanced. This only applies if the'
                                 ' --train option is used.')
