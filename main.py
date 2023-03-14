import argparse
import logging

from utils import parser_config, setup_logging, set_random_seed, exit_handler

from numba.core.errors import NumbaWarning
import warnings


def main():
    warnings.simplefilter('ignore', category=NumbaWarning)
    logging.getLogger('numba').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    arguments = [
        ("rawcreate", create_raw_dataset, "Create Raw Dataset"),
        ("preprocess", preprocessing, "Preprocess samples - cleaning/filtering of invalid d ata"),
        ("split", split_dataset, "Splitting train dataset into train/dev folds"),
        ("prepare_input", prepare_input, "Preparing input for train process"),
        ("train", train, "Train currently selected model"),
        ("test", test, "Predict on test data then get output for submitting")
    ]
    for arg, _, description in arguments:
        parser.add_argument('--{}'.format(arg), action ='store_true', help=description)

    params = parser.parse_args()
    args = parser_config("config.json")
    
    setup_logging(args)
    set_random_seed(args)
    for arg, fun, _ in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            logging.info("Performing {} operation..".format(arg))
            fun(args)
if __name__ == "__main__":
    main()