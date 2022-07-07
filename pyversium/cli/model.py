import argparse
import json
import logging
import os
import re
import sys

import pandas as pd

from .parsing import get_config, ParserCollection, parser_add_log_options, parser_add_collector_options
from .schema import MODEL_CONFIG_SCHEMA_PATH
from ..collect import Collector
from ..feature_selection.inferred_selector import InferredFeatureSelector
from ..insights.evaluation import BinaryModelInsights
from ..modeling import BinaryModelFactory, BinaryModel
from ..postprocessing import QuantilePostprocessor
from ..utils.io import ModelPaths, OutputFileGenerator
from ..utils.logging import setup_logging

DEFAULT_CONFIG = {}

logging.basicConfig()
# Set logger name to begin with pyversium if this is the main program. This will ensure that everything in this file gets logged.
logger = logging.getLogger('pyversium.__main__(model)' if __name__ == '__main__' else __name__)


def get_parser(defaults=None):
    parser = argparse.ArgumentParser(prog="model", add_help=True)

    # Parse options common to all subparsers
    global_parser = argparse.ArgumentParser(add_help=False)

    # Positional arguments for all parsers
    global_parser.add_argument("-m", "--model-dir", action="store", type=str, default=None,
                               help="Directory to store models and metadata. This MUST be provided either in the command line arguments "
                                    "or config file.")
    global_parser.add_argument("-i", "--input", action="store", type=str, default=None, help="Input file for modeling or scoring.")

    global_parser.add_argument("-c", "--config", action="store", type=str, default=None, metavar="CONFIG FILE",
                               help="Path to configuration file.")

    parser_add_log_options(global_parser)
    parser_add_collector_options(global_parser, base=True, chunk=False, label=False, balance=False)

    if defaults is not None:
        global_parser.set_defaults(**defaults)

    # Create subparsers
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Train subparser and options
    train_parser = subparsers.add_parser("train", parents=[global_parser], add_help=True,
                                         help="Train a model")

    score_parser = subparsers.add_parser("score", parents=[global_parser], add_help=True, help="Score a file using a trained model.")

    train_parser.add_argument("--include-columns", action="store", type=str, nargs="*", default=[],
                              help="Columns to include as features (whitespace delimited). If provided, only columns in this list will be used.")

    train_parser.add_argument("--exclude-columns", action="store", type=str, nargs="*", default=[],
                              help="Columns to exclude as features. If provided, matching columns will not be used in modeling.")

    train_parser.add_argument("--regex-include-columns", action="store", type=str, nargs="*", default=[],
                              help="Use regex expressions to include any features that match the pattern.")

    train_parser.add_argument("--regex-exclude-columns", action="store", type=str, nargs="*", default=[],
                              help="Use regex expressions to exclude any features that match the pattern.")

    train_parser.add_argument("--num-opt", action="store", type=int, default=None, metavar="NUM OPT ROUNDS",
                              help="Number of optimization rounds to perform for model tuning.")

    parser_add_collector_options(train_parser, base=False, chunk=False, label=True, balance=False)

    # Score subparser and options
    score_parser.add_argument("-o", "--output", action="store", type=str, default=None,
                              help="File to write output results to. Defaults to stdout. \n" + OutputFileGenerator.help())

    parser_add_collector_options(score_parser, base=False, chunk=True, label=False, balance=False)

    return ParserCollection(parser, [train_parser, score_parser])


def calc_include_exclude_columns(columns: list[str],
                                 include: list[str] = (),
                                 exclude: list[str] = (),
                                 include_regex: list[str] = (),
                                 exclude_regex: list[str] = ()
                                 ) -> (list[str], list[str]):
    regex_inc_set = set()
    regex_ex_set = set()

    for pattern in include_regex:
        regex_inc_set |= set(filter(lambda x: re.fullmatch(pattern, x), columns))

    for pattern in exclude_regex:
        regex_inc_set |= set(filter(lambda x: re.match(pattern, x), columns))

    include = set(include) | regex_inc_set
    exclude = set(exclude) | regex_ex_set

    # Use all columns if we have not specified any.
    if not include:
        include = set(columns)
    else:
        include &= set(columns)

    return list(include), list(exclude)


def train(config):
    model_dir = config['model_dir']
    model_paths = ModelPaths(model_dir)
    if not config['label']:
        raise ValueError("A label column is required when training a model. Use -l or --label to provide the label column.")

    if not config.get('input', None):
        raise ValueError("No `input` provided via the command line nor a config file.")

    collector = Collector(missing_values=config.get('missing_values', None),
                          required_fields=config.get('required_fields', []),
                          label=config['label'],
                          label_positive_value=config['label_positive_value'],
                          consolidate_missing=True)

    data = collector.collect(config['input'],
                             delimiter=config['delimiter'],
                             has_label=True,
                             chunksize=None,
                             header=config['header'])

    logger.info(f"Label Value Counts:\n{data[config['label']].value_counts()}")

    include, exclude = calc_include_exclude_columns(data.columns,
                                                    include=config['include_columns'],
                                                    exclude=config['exclude_columns'],
                                                    include_regex=config['regex_include_columns'],
                                                    exclude_regex=config['regex_exclude_columns'])

    logger.debug(f"\ninclude: {include}\nexclude: {exclude}")

    feature_selector = InferredFeatureSelector(include_columns=include, exclude_columns=exclude)
    label = data[collector.label].astype(int)
    post = QuantilePostprocessor()

    model_factory = BinaryModelFactory(feature_selector,
                                       postprocessor=post,
                                       test_size=0.2,
                                       random_state=123456,
                                       n_optimization_rounds=config['num_opt']
                                       )
    model_factory.optimize(data, label)
    model_factory.save(model_paths)

    model_insights = BinaryModelInsights(model_factory)
    model_insights.fit(data, label)
    model_insights.generate_report(model_paths)


def score(config):
    model_dir = os.path.abspath(config['model_dir'])
    model_paths = ModelPaths(model_dir)

    if not config.get('input', None):
        raise ValueError("No `input` provided via the command line nor a config file.")

    model_dir = config.pop('model_dir')

    logger.debug(f"Model directory is {model_dir}")

    input_file = config.pop('input')

    logger.debug(f"Input file is {input_file}")

    output_file = config.pop('output')

    if isinstance(output_file, str):
        logger.debug(f"Output file is {output_file}")
    else:
        logger.debug(f"No output file provided. Defaulting to stdout")

    output_file_gen = OutputFileGenerator(sys.stdout if output_file is None else output_file, input_file=input_file)

    logger.info("Begin reading input file.")

    collector = Collector(
        # missing_values=config['missing_values'],
        # required_fields=config['required_fields'],
        # label=config['label'],
        # label_positive_value=config['label_positive_value'],
        consolidate_missing=True)

    data = collector.collect(input_file,
                             delimiter=config['delimiter'],
                             has_label=False,
                             chunksize=config['chunksize'],
                             header=config['header'])

    logger.info("Finished reading input file.")
    logger.info("Begin loading saved model.")

    model = BinaryModel.load(model_paths)

    logger.info("Finished loading model.")
    logger.debug(f"Model type is: {type(model)}")
    logger.debug("Model architecture is:\n {}".format(model.__repr__().replace('\n', '\r')))

    if isinstance(data, pd.DataFrame):
        data = [data]

    logger.info("Begin scoring data.")

    for i, chunk in enumerate(data):
        output_file = next(output_file_gen)
        scores = model.score(chunk)

        try:
            # Try to convert to all integer scores.
            new_dtype = pd.Int64Dtype()
            scores = pd.Series(scores, name='Score', dtype=new_dtype)

            logger.debug(f"Successfully converted scores to dtype {new_dtype}.")

        except TypeError:

            logger.debug(f"Could not convert scores to nullable integer type. Keeping as {scores.dtype}")

            scores = pd.Series(scores, name='Score')

        if isinstance(output_file, str):
            logger.info(f"Saving scores to {output_file}")
        else:
            logger.info(f"Printing scores to stdout.")

        # Overwrite file if it is the first iteration or we are generating multiple output file names. Otherwise append.
        mode = 'w+' if i == 0 or output_file_gen.format_output else 'a+'
        scores.to_csv(output_file, sep=config['delimiter'], index=False, mode=mode)

    logger.info("Finished scoring data.")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    with open(MODEL_CONFIG_SCHEMA_PATH) as f:
        model_config_schema = json.load(f)

    parser = get_parser()
    config = get_config(args,
                        parser=parser,
                        config_schema=model_config_schema,
                        subparser_dest="cmd",
                        all_cmd_names=["train", "score"])

    # Get additional packages to log
    package_logging = config.get('log_packages')
    if 'pyversium' not in package_logging:
        package_logging += ['pyversium']

    model_dir = config.get('model_dir', None)
    if not model_dir:
        raise ValueError("`model_dir` was not provided in a config file nor via the command line.")

    model_paths = ModelPaths(model_dir)

    cmd = config['cmd']
    msg = f"No cmd found matching `{cmd}`. Exiting program."
    level = logging.getLevelName('CRITICAL')
    cmd_func = lambda x: quit(msg)
    default_log_file_name = model_paths.model_log_file

    if config['cmd'] == 'train':
        default_log_file_name = model_paths.train_log_file
        msg = "Calling `train` CLI command."
        level = logging.getLevelName('INFO')
        cmd_func = train

    elif config['cmd'] == 'score':
        default_log_file_name = model_paths.score_log_file
        msg = "Calling `score` CLI command."
        level = logging.getLevelName('INFO')
        cmd_func = score

    log_files = config.pop('log_file', None)
    if log_files:
        log_files = [log_files]
    else:
        log_files = []
    log_files += [default_log_file_name]

    setup_logging(level=config.pop('log_level', 'ERROR'), log_file=log_files,
                  packages=package_logging, file_mode='w+' if config['overwrite_logs'] else 'a+')

    logger.debug(f"config is: {config}")

    logger.log(level=level, msg=msg)
    cmd_func(config)


if __name__ == '__main__':
    main()
