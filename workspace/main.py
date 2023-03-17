import argparse
import logging

from numba.core.errors import NumbaWarning
import warnings
from create_dataset import create_datset

def main():
    warnings.simplefilter('ignore', category=NumbaWarning)
    logging.getLogger('numba').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    arguments = [
        "create_dataset", create_dataset, "create dataset"
    ]
    for arg, _, description in arguments:
        parser.add_argument('--{}'.format(arg), action ='store_true', help=description)

    params = parser.parse_args()

    print(params)


if __name__ == "__main__":
    main()