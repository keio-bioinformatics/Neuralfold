import numpy as np
import sys
import SStruct
import Inference
import Evaluate
import Train
import Test
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
import argparse

def train(args):
    print("start training...")
    sstruct = SStruct.SStruct(args.train_file)
    name_set, seq_set, structure_set = sstruct.load_FASTA()

    sstruct = SStruct.SStruct(args.test_file)
    name_set_test, seq_set_test, structure_set_test = sstruct.load_FASTA()

    train = Train.Train(seq_set, structure_set, seq_set_test, structure_set_test)
    model = train.train()
    serializers.save_npz("NEURALfold_params.data", model)

def test(args):
    print("start testing...")
    sstruct = SStruct.SStruct(args.test_file)
    name_set_test, seq_set_test, structure_set_test = sstruct.load_FASTA()
    test = Test.Test(seq_set_test)
    predicted_structure_set = test.test()
    evaluate = Evaluate.Evaluate(predicted_structure_set , structure_set_test)
    Sensitivity, PPV, F_value = evaluate.getscore()
    print(Sensitivity, PPV, F_value)

def main():
    parser = argparse.ArgumentParser(
                prog='Neuralfold',
                usage='prediction of RNA secondary structures',
                description='description',
                epilog='end',
                add_help=True,
                )

    subparser = parser.add_subparsers(
                title='SubCommands',
                description='SubCommands descript',
                help='SubCommand help')

    # add subparser for training
    parser_training = subparser.add_parser('train', help='training RNA secondary structures')
    parser_training.add_argument('train_file', help = 'FASTA file for training',
                        type=argparse.FileType('r'))
    parser_training.add_argument('test_file', help = 'FASTA file for test',
                        type=argparse.FileType('r'))
    parser_training.set_defaults(func = train)


    # add subparser for test
    parser_test = subparser.add_parser('test', help='test secondary structures')
    parser_test.add_argument('test_file', help = 'FASTA file for test',
                        type=argparse.FileType('r'))
    parser_test.set_defaults(func = test)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
