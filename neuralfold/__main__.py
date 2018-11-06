#!/usr/bin/env python

import argparse
import os.path
import sys

from .predict import Predict
#from . import Evaluate
from .Train import Train


def main():
    parser = argparse.ArgumentParser(
        description='NeuralFold: Direct inference of base-pairing probabilities with neural networks improves RNA secondary structure prediction with pseudoknots.',
        add_help=True,
    )
    subparser = parser.add_subparsers(title='Sub-commands')
    parser.set_defaults(func = None)
    Train.add_args(subparser)
    Predict.add_args(subparser)
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
    else:
        args.func(args)

if __name__ == '__main__':
    main()
