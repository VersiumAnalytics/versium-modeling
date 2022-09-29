"""append.py: CLI entry point for append tasks."""
import argparse
import logging
import os
import sys

import pandas as pd

from ._io import get_output_generator
from ._parsing import (parser_add_log_options,
                       parser_add_collector_options,
                       parser_add_output_options,
                       parser_add_input_options,
                       get_config, ParserCollection,
                       GLOBAL_EPILOG,
                       WhitespaceHelpFormatter)
from ..collect import Collector, QueryClient
from ..utils import setup_logging, filter_params

DEFAULT_CONFIG = {'query_configs': ()}

logging.basicConfig()
# Set logger name to begin with vmod if this is the main program. This will ensure that everything in this file gets logged.
logger = logging.getLogger('vmod.__main__(append)' if __name__ == '__main__' else __name__)

# Use carriage returns instead of newlines. These will be preserved as linebreaks in the help output while newlines are converted to
# single spaces.
EPILOG = """
        Balancing Columns:\r
        If we intend to use our appended data for modeling purposes, then we need to avoid introducing data leakage into our training
        data. This can be particularly problematic when we have converter/non-converter as the positive and negative class in our data.
        A common issue is that we will have more accurate or more complete information for converters vs non-converters, resulting in
        better append fill rates for the positive class. This results in data leakage and poor model performance since most supervised
        learning algorithms will pick up on this pattern instead of the true underlying signal that we wish to model against.
"""

TRAIN_EPILOG = """
        The `train` subcommand has two purposes. First, it allows us to specify a configuration in the config file that will be used when the
        `train` subcommand is invoked. This can be used to apply different append behavior to data that will be used for model training
        and data that will be used for model scoring. For example, we probably want to give different output file names to our training and
        scoring data without having to explicitly pass the `--output` option on the CLI every time. For instructions on how to create a
        separate `train` configuration in the config file, see the README.
        
        The second purpose of the `train` subcommand is to perform preprocessing that is specific to model training data. For example,
        column balancing is performed to help remove spurious correlation between the label column and append fill rates.
"""

SCORE_EPILOG = """
        The `score` subcommand is mostly included for consistency with the the `model` CLI tool. It performs a normal append with the same
        behavior as if a subcommand were omitted. The one difference is that the "score" configuration from the config file will be used if
        the `score` subcommand is invoked.
"""


def get_parser(defaults: dict | None = None):
    # Parse options common to all subparsers
    global_parser = argparse.ArgumentParser(add_help=False, epilog=EPILOG, formatter_class=WhitespaceHelpFormatter)

    parser_add_input_options(global_parser)
    parser_add_output_options(global_parser)

    global_parser.add_argument("-c", "--config", action="store", type=str, default=None, metavar="CONFIG FILE",
                               help="Path to configuration file.")

    parser_add_log_options(global_parser)
    parser_add_collector_options(global_parser, base=True, chunk=True, label=True, balance=True)

    if defaults is not None:
        global_parser.set_defaults(**defaults)

    # Use carriage returns instead of newlines. These will be preserved as linebreaks in the help output while newlines are converted to
    # single spaces.
    global_epilog = EPILOG + "\r\r" + GLOBAL_EPILOG
    train_epilog = EPILOG + "\r\r" + TRAIN_EPILOG
    score_epilog = EPILOG + "\r\r" + SCORE_EPILOG
    # Create main parser. Will inherit from global
    parser = argparse.ArgumentParser(prog="append", add_help=True, parents=[global_parser], epilog=global_epilog,
                                     formatter_class=WhitespaceHelpFormatter)

    # Create subparsers
    subparsers = parser.add_subparsers(dest="cmd", required=False)
    # Train subparser and options
    subparsers.add_parser("train", parents=[global_parser], add_help=True,
                          help="Use options for appending to model training data.", epilog=train_epilog)

    subparsers.add_parser("score", parents=[global_parser], add_help=True,
                          help="Use options for appending to model scoring data.", epilog=score_epilog)

    return ParserCollection(parser)


def collector_factory(config: dict) -> Collector:
    api_queries = []
    query_configs = config.pop('query_configs', [])
    for q_cfg in query_configs:
        if 'url' not in q_cfg:
            raise ValueError('Each query config must have an api endpoint defined by `url`')
        else:
            url = q_cfg.pop('url')
        q_cfg_filtered = filter_params(q_cfg, QueryClient)
        if len(q_cfg_filtered) != len(q_cfg):
            raise ValueError(f"Unknown query configurations given: {set(q_cfg) - set(q_cfg_filtered)}")
        api_queries.append(QueryClient(url, **q_cfg))

    # Need balance_diff_tol to be a float instead of None for the constructor
    balance_diff_tol = config.pop('balance_diff_tol')
    balance_diff_tol = balance_diff_tol if balance_diff_tol is not None else 0.1

    return Collector(balance_diff_tol=balance_diff_tol,
                     api_queries=api_queries,
                     missing_values=config['missing_values'],
                     required_fields=config['required_fields'],
                     label=config['label'],
                     balance_fields=config['balance_fields'],
                     label_positive_value=config['label_positive_value'],
                     correct_label_field=False,
                     consolidate_missing=True)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    config = get_config(args,
                        parser=parser,
                        subparser_dest="cmd",
                        all_cmd_names=["train", "score"]
                        )

    # Get additional packages to log
    package_logging = config.pop('log_packages')

    if 'vmod' not in package_logging:
        package_logging += ['vmod']

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

    delimiter = config["delimiter"]
    header = config["header"]
    chunksize = config["chunksize"]
    chunkstart = config["chunkstart"]

    train = config.get("cmd", "") == "train"

    collector = collector_factory(config)

    output_gen = get_output_generator(output_file, input_file=input_file, chunkstart=chunkstart, chunksize=chunksize, truncate=True,
                                      delimiter=delimiter, has_header=True)

    logger.info("Begin collecting data.")
    data, field_names = collector.collect(input_file, delimiter=delimiter, has_label=train, chunksize=chunksize, chunkstart=chunkstart,
                                          header=header)

    if isinstance(data, pd.DataFrame):
        data = [data]

    for i, chunk in enumerate(data):
        if chunksize:
            logger.info(f"Processed chunk {i + chunkstart}.")
        output, mode = next(output_gen)
        header = False if mode.lower().startswith("a") else True
        chunk.to_csv(output, sep=delimiter, index=False, mode=mode, header=header)


if __name__ == '__main__':
    main()
