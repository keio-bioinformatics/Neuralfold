#!/usr/bin/env python

import sys
import os.path
#from . import Evaluate
from .Train import Train
from .Test import Test
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog='Neuralfold',
        usage=os.path.basename(sys.argv[0]), #'neuralfold',
        description='description',
        epilog='end',
        add_help=True,
    )

    subparser = parser.add_subparsers(
        title='SubCommands',
        description='SubCommands descript',
        help='SubCommand help')

    Train.parse_args(subparser)
    Test.parse_args(subparser)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
