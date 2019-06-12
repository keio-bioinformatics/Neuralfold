#!/usr/bin/env python

from argparse import ArgumentParser
import os.path
import sys

from .predict import Predict
from .train import Train
from .optimize import Optimize


def main():
    parser = ArgumentParser(
        description='NeuralFold: Direct inference of base-pairing probabilities '
            'with neural networks improves RNA secondary structure prediction with pseudoknots.',
        fromfile_prefix_chars='@',
        add_help=True,
    )
    subparser = parser.add_subparsers(title='Sub-commands')
    parser.set_defaults(func = lambda args: parser.print_help())
    Train.add_args(subparser)
    Predict.add_args(subparser)
    Optimize.add_args(subparser)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
