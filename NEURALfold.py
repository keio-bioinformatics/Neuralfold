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
    train = Train.Train(args)
    model = train.train()
    serializers.save_npz("NEURALfold_params.data", model)

def test(args):
    print("start testing...")
    test = Test.Test(args)
    Sensitivity, PPV, F_value = test.test()
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
    parser_training.add_argument('-t','--test_file', help = 'FASTA file for test',
                        type=argparse.FileType('r'))
    parser_training.add_argument('-p','--Parameters', help = 'Parameter file',
                        type=argparse.FileType('r'))
    parser_training.add_argument('-o','--Optimizers', help = 'Optimizer file',
                        type=argparse.FileType('r'))
    parser_training.add_argument('-i','--iteration', help = 'number of iteration',
                        type=int,default=1)

    # neural networks architecture
    parser_training.add_argument('-H1','--hidden_insideoutside', help = 'hidden layer nodes for inside outside',
                        type=int,default=80)
    parser_training.add_argument('-H2','--hidden2_insideoutside', help = 'hidden layer nodes2 for inside outside',
                        type=int)
    parser_training.add_argument('-h1','--hidden_marge', help = 'hidden layer nodes for marge phase',
                        type=int,default=80)
    parser_training.add_argument('-h2','--hidden2_marge', help = 'hidden layer nodes2 for marge phase',
                        type=int)
    parser_training.add_argument('-f','--feature', help = 'feature length',
                        type=int,default=80)
    parser_training.add_argument('-a','--activation_function', help = 'activation function for output',
                        type=str,default="softmax")
    parser_training.add_argument('-l','--learning_model', help = 'learning_model',
                        type=str,default="recursive")

    # training option
    parser_training.add_argument('-m','--max_margin', help = 'use maxmargin',
                        action = 'store_true')

    parser_training.set_defaults(func = train)


    # add subparser for test
    parser_test = subparser.add_parser('test', help='test secondary structures')
    parser_test.add_argument('test_file', help = 'FASTA file for test',
                        type=argparse.FileType('r'))
    parser_test.add_argument('-p','--Parameters', help = 'Parameter file',
                        type=argparse.FileType('r'))
    parser_test.set_defaults(func = test)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
