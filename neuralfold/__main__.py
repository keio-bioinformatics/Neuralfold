#!/usr/bin/env python

import sys
import os.path
#from . import Evaluate
from .Train import Train
from .Test import Test
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='NeuralFold: Direct inference of base-pairing probabilities with neural networks improves RNA secondary structure prediction with pseudoknots.',
        add_help=True,
    )
    subparser = parser.add_subparsers(title='Subcommands')
    parser.set_defaults(func = None)
    Train.add_args(subparser)
    Test.add_args(subparser)
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
    else:
        args.func(args)

if __name__ == '__main__':
    main()