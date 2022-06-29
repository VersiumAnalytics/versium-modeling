import argparse
import json
import logging
import os
import sys
import textwrap

import pandas as pd

from .parsing import parser_add_log_options, parser_add_collector_options, get_config, ParserCollection
from .schema import APPEND_CONFIG_SCHEMA_PATH
from ..collect import Collector, QueryClient
from ..utils import OutputFileGenerator
from ..utils import setup_logging

DEFAULT_CONFIG = {'query_configs': ()}

logging.basicConfig()
# Set logger name to begin with pyversium if this is the main program. This will ensure that everything in this file gets logged.
logger = logging.getLogger('pyversium.__main__(append)' if __name__ == '__main__' else __name__)


def get_parser_orig():
    parser = argparse.ArgumentParser(add_help=True)

    # Options needed for file reading and writing
    parser.add_argument('-i', '--input', action='store', type=str)
    parser.add_argument('-o', '--output', action='store', type=str, default=None,
                        help='File to write output results to. Defaults to stdout. If the --write-chunksize argument is used, then'
                             'numbering of files can be controlled by specifying the path in the format "my_dir/my_file_{005}.txt'
                             'where the number of digits signifies zero-padding and the number itself indicates what number to start at.'
                             'If --input-file is provided, you can use $@ to have the input filename injected into the output file name'
                             '(e.g. my_output_dir/$@_appended_{000}.txt)')

    # CLI only arguments
    parser.add_argument('-t', '--train', action='store_true', help='Indicates this data is for training a _model and should be handled'
                                                                   'appropriately.')

    parser.add_argument('--train-input', action='store', type=str, default=None,
                        help='Same as --input, but this will be used instead when the --train option is used. Falls back to'
                             '--input if not provided. Used to help avoid mixing training and scoring data accidentally.')

    parser.add_argument('--train-output', action='store', type=str, default=None,
                        help='Same as --output, but this will be used instead when the --train option is used. Falls back to'
                             '--output if not provided. Used to help avoid mixing training and scoring data accidentally.')

    # Additional program arguments
    parser.add_argument('-c', '--config', action='store', type=str, default=None, metavar='CONFIG FILE',
                        help='Path to configuration file.')

    parser_add_log_options(parser)
    parser_add_collector_options(parser)

    # parser.add_argument('--resume', action='store', type=int, metavar='CHUNK', help='Resume appending from CHUNK. File chunks are enumerated starting at index 0.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Force overwriting of files if a file already exists. By default, if file already exists, then numbers will be'
                             'added to the filename to give it a unique name.')

    epilog_text = textwrap.dedent('''
        Balancing Columns:
            If we intend to use our appended data for modeling purposes, then we need to avoid introducing data leakage into our training data.
            This can be particularly problematic when we have converter/non-converter as the positive and negative class in our data. A common
            issue is that we will have more accurate or more complete information for converters vs non-converters, resulting in better append fill rates
            for the positive class. This results in data leakage and poor _model performance since most supervised learning algorithms will pick up on this
            pattern instead of the true underlying signal that we wish to _model against.
    ''')

    return ParserCollection(parser)


def get_parser(defaults=None):
    # Parse options common to all subparsers
    global_parser = argparse.ArgumentParser(add_help=False)

    global_parser.add_argument("-i", "--input", action="store", type=str, default=None, help="Input file.")

    global_parser.add_argument('-o', '--output', action='store', type=str, default=None,
                               help='Output file')

    global_parser.add_argument("-c", "--config", action="store", type=str, default=None, metavar="CONFIG FILE",
                               help="Path to configuration file.")

    parser_add_log_options(global_parser)
    parser_add_collector_options(global_parser, base=True, chunk=True, label=True, balance=True)

    if defaults is not None:
        global_parser.set_defaults(**defaults)

    # Create main parser. Will inherit from global

    parser = argparse.ArgumentParser(prog="append", add_help=True, parents=[global_parser])

    # Create subparsers
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    subparsers.add_parser("score", parents=[global_parser], add_help=True,
                          help="Use options for appending to model scoring data.")

    # Train subparser and options
    subparsers.add_parser("train", parents=[global_parser], add_help=True,
                          help="Use options for appending to model training data.")

    # parser.add_argument('--resume', action='store', type=int, metavar='CHUNK', help='Resume appending from CHUNK. File chunks are enumerated starting at index 0.')
    # parser.add_argument('--overwrite', action='store_true',
    #                    help='Force overwriting of files if a file already exists. By default, if file already exists, then numbers will be'
    #                         'added to the filename to give it a unique name.')

    epilog_text = textwrap.dedent('''
        Balancing Columns:
            If we intend to use our appended data for modeling purposes, then we need to avoid introducing data leakage into our training data.
            This can be particularly problematic when we have converter/non-converter as the positive and negative class in our data. A common
            issue is that we will have more accurate or more complete information for converters vs non-converters, resulting in better append fill rates
            for the positive class. This results in data leakage and poor _model performance since most supervised learning algorithms will pick up on this
            pattern instead of the true underlying signal that we wish to _model against.
    ''')

    return ParserCollection(parser)


def collector_factory(config: dict) -> Collector:
    api_queries = []
    query_configs = config.pop('query_configs', [])
    for q_cfg in query_configs:
        if 'url' not in q_cfg:
            raise ValueError('Each query config must have an api endpoint defined by `url`')
        else:
            url = q_cfg.pop('url')
        api_queries.append(QueryClient(url, **q_cfg))

    # Need balance_diff_tol to be a float instead of None for the constructor
    balance_diff_tol = config.pop('balance_diff_tol')
    balance_diff_tol = balance_diff_tol if balance_diff_tol is not None else 0.1

    return Collector(balance_diff_tol=balance_diff_tol,
                     api_queries=api_queries,
                     missing_values=config['missing_values'],
                     required_fields=config['required_fields'],
                     label=config['label'],
                     label_positive_value=config['label_positive_value'],
                     consolidate_missing=True)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    with open(APPEND_CONFIG_SCHEMA_PATH) as f:
        append_config_schema = json.load(f)

    parser = get_parser()
    config = get_config(args,
                        parser=parser,
                        config_schema=append_config_schema,
                        subparser_dest="cmd",
                        all_cmd_names=["train", "score"]
                        )

    # Get additional packages to log
    package_logging = config.pop('log_packages')

    if 'pyversium' not in package_logging:
        package_logging += ['pyversium']

    log_files = config.pop('log_file', [])
    if log_files:
        log_files = [log_files]

    setup_logging(level=config.pop('log_level', 'ERROR'), log_file=log_files,
                  packages=package_logging, file_mode='w+' if config['overwrite_logs'] else 'a+')

    logger.debug(f"config is: {config}")

    input_file = config.pop('input', None)

    if input_file is None:
        raise IOError(f"`input` not provided via command line nor config file.")
    elif not os.path.exists(input_file):
        raise IOError(f'`input` {input_file} does not exist')

    input_file = os.path.abspath(input_file)
    output_file = config.pop('output', None)

    output_file_gen = OutputFileGenerator(sys.stdout if output_file is None else output_file, input_file=input_file)

    delimiter = config.pop('delimiter', '\t')
    header = config.pop('header', None)
    # resume = config.pop('resume', None)
    overwrite = config.pop('overwrite', True)
    chunksize = config.pop('chunksize')
    train = config.get("cmd", "") == "train"

    collector = collector_factory(config)

    logger.info("Begin collecting data.")
    data = collector.collect(input_file, delimiter=delimiter, has_label=train, chunksize=chunksize, header=header)

    if isinstance(data, pd.DataFrame):
        data = [data]

    for i, chunk in enumerate(data):
        if chunksize:
            logger.info(f"Processed chunk {i + 1}.")
        output_file = next(output_file_gen)
        chunk.to_csv(output_file, sep=delimiter, index=False, mode='w' if overwrite else 'a')


if __name__ == '__main__':
    main()
