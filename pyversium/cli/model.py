import argparse
import logging
import os
import re
import sys

import pandas as pd

from ._io import get_output_generator
from ._parsing import (get_config,
                       ParserCollection,
                       parser_add_log_options,
                       parser_add_collector_options,
                       parser_add_output_options,
                       parser_add_input_options,
                       parser_add_include_exclude_options)
from ..collect import Collector
from ..feature_selection.inferred_selector import InferredFeatureSelector
from ..insights.evaluation import BinaryModelInsights
from ..modeling import BinaryModelFactory, BinaryModel
from ..postprocessing import QuantilePostprocessor
from ..utils.io import ModelPaths
from ..utils.logging import setup_logging

DEFAULT_CONFIG = {}

logging.basicConfig()
# Set logger name to begin with pyversium if this is the main program. This will ensure that everything in this file gets logged.
logger = logging.getLogger('pyversium.__main__(model)' if __name__ == '__main__' else __name__)


def get_parser(defaults: dict | None = None):
    parser = argparse.ArgumentParser(prog="model", add_help=True)

    # Parse options common to all subparsers
    global_parser = argparse.ArgumentParser(add_help=False)

    # Positional arguments for all parsers
    global_parser.add_argument("-m", "--model-dir", action="store", type=str, default=None,
                               help="Directory to store models and metadata. This MUST be provided either in the command line arguments "
                                    "or config file.")

    parser_add_input_options(global_parser)

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

    parser_add_include_exclude_options(train_parser, purpose_string="be used as features in modeling.")

    train_parser.add_argument("--num-opt", action="store", type=int, default=None, metavar="NUM OPT ROUNDS",
                              help="Number of optimization rounds to perform for model tuning.")

    parser_add_collector_options(train_parser, base=False, chunk=False, label=True, balance=False)

    # Score subparser and options
    parser_add_output_options(score_parser)
    parser_add_include_exclude_options(score_parser, purpose_string="be output with scores.")
    parser_add_collector_options(score_parser, base=False, chunk=True, label=False, balance=False)
    score_parser.add_argument("--score-field-name", action="store", type=str, default="Score",
                              help="Field name to use for model scores in the output. (default: %(default)s)")

    return ParserCollection(parser, [train_parser, score_parser])


def calc_include_exclude_fields(fields: list[str],
                                include: list[str] = (),
                                exclude: list[str] = (),
                                include_regex: list[str] = (),
                                exclude_regex: list[str] = ()
                                ) -> (list[str], list[str], list[str]):
    """Filter a list of field names using inclusion and exclusion rules

    Parameters
    ----------
    fields : input fields to filter
    include : explicit list of fields to include
    exclude : explicit list of fields to exclude
    include_regex : Regex patterns for including fields that match any of the patterns
    exclude_regex : Regex patterns for excluding fields that match any of the patterns

    Returns
    -------
    Tuple of filtered fields, include fields, and exclude fields

    """
    regex_inc_set = set()
    regex_ex_set = set()

    for pattern in include_regex:
        regex_inc_set |= set(filter(lambda x: re.fullmatch(pattern, x), fields))

    for pattern in exclude_regex:
        regex_ex_set |= set(filter(lambda x: re.match(pattern, x), fields))

    include_set = set(include) | regex_inc_set
    exclude_set = set(exclude) | regex_ex_set

    # Use all fields if we have not specified any.
    if not include_set:
        include_set = set(fields)
    else:
        include_set &= set(fields)

    filtered_set = include_set - exclude_set
    filtered_fields = [f for f in fields if f in filtered_set]
    return filtered_fields, list(include_set), list(exclude_set)


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

    data, field_names = collector.collect(config['input'],
                                          delimiter=config['delimiter'],
                                          has_label=True,
                                          chunksize=None,
                                          header=config['header'])

    logger.info(f"Label Value Counts:\n{data[config['label']].value_counts()}")

    filtered_fields, include, exclude = calc_include_exclude_fields(field_names,
                                                                    include=config['include_fields'],
                                                                    exclude=config['exclude_fields'],
                                                                    include_regex=config['regex_include_fields'],
                                                                    exclude_regex=config['regex_exclude_fields'])

    logger.debug(f"\ninclude: {include}\nexclude: {exclude}")

    if not filtered_fields:
        raise ValueError("After taking `include` and `exclude` fields into account, there are no fields left for modeling.")

    feature_selector = InferredFeatureSelector(include_fields=include, exclude_fields=exclude)
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
    chunksize = config["chunksize"]
    chunkstart = config["chunkstart"]
    delimiter = config["delimiter"]
    header = config["header"]

    logger.info("Begin reading input file.")

    collector = Collector(
        missing_values=config['missing_values'],
        required_fields=config['required_fields'],
        consolidate_missing=True)

    data, field_names = collector.collect(input_file,
                                          delimiter=delimiter,
                                          has_label=False,
                                          chunksize=chunksize,
                                          chunkstart=chunkstart,
                                          header=header)

    logger.info("Finished reading input file.")

    extra_output_cols, include, exclude = calc_include_exclude_fields(fields=field_names,
                                                                      include=config['include_fields'],
                                                                      exclude=config['exclude_fields'],
                                                                      include_regex=config['regex_include_fields'],
                                                                      exclude_regex=config['regex_exclude_fields'])

    logger.info("Begin loading saved model.")
    model = BinaryModel.load(model_paths)

    logger.info("Finished loading model.")
    logger.debug(f"Model type is: {type(model)}")
    logger.debug("Model architecture is:\n {}".format(model.__repr__()))

    if isinstance(data, pd.DataFrame):
        data = [data]

    logger.info("Begin scoring data.")

    output_gen = get_output_generator(output_file, input_file=input_file, chunkstart=chunkstart, chunksize=chunksize, truncate=True,
                                      delimiter=delimiter, has_header=True)

    score_field_name = config.get("score_field_name", "Score")
    for i, chunk in enumerate(data):
        output_file, mode = next(output_gen)
        scores = model.score(chunk)

        try:
            # Try to convert to all integer scores.
            new_dtype = pd.Int64Dtype()
            scores = pd.Series(scores, name=score_field_name, dtype=new_dtype)
            logger.debug(f"Successfully converted scores to dtype {new_dtype}.")

        except TypeError:

            logger.debug(f"Could not convert scores to nullable integer type. Keeping as {scores.dtype}")

            scores = pd.Series(scores, name=score_field_name)

        if isinstance(output_file, str):
            logger.info(f"Saving scores to {output_file}")
        else:
            logger.info(f"Printing scores to stdout.")

        if extra_output_cols:
            output = chunk[extra_output_cols].reset_index(drop=True).join(scores, rsuffix="_SCORED")
        else:
            output = scores

        header = False if mode.lower().startswith("a") else True
        output.to_csv(output_file, sep=config['delimiter'], index=False, mode=mode, header=header)

    logger.info("Finished scoring data.")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    config = get_config(args,
                        parser=parser,
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
