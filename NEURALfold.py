import numpy as np
import sys
import SStruct
import Inference
import Evaluate
import Train
import Test
import argparse

def train(args):
    print("start training...")
    Train.Train(args).train()

def test(args):
    print("start testing...")
    Sensitivity, PPV, F_value = Test.Test(args).test()


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
    parser_training.add_argument('train_file', help = 'FASTA or BPseq file for training', nargs='+',
                                 type=argparse.FileType('r'))
    parser_training.add_argument('-t','--test_file', help = 'FASTA file for test', nargs='+',
                                 type=argparse.FileType('r'))
    parser_training.add_argument('--init-parameters', help = 'Initial parameter file',
                                 type=str)
    parser_training.add_argument('-p','--parameters', help = 'Output parameter file',
                                 type=str, default="NEURALfold_parameters")
    parser_training.add_argument('--init-optimizers', help = 'Initial optimizer state file',
                                 type=argparse.FileType('r'))
    parser_training.add_argument('-o','--optimizers', help = 'Optimizer state file',
                                 type=str, default="state.npz")
    parser_training.add_argument('-i','--iteration', help = 'number of iteration',
                                 type=int, default=1)
    parser_training.add_argument('-bp','--bpseq', help = 'use bpseq',
                                 action = 'store_true')

    # neural networks architecture
    parser_training.add_argument('-l','--learning_model',
                                 help = 'learning_model',
                                 type=str, default="deepnet")

    parser_training.add_argument('-H1','--hidden_insideoutside',
                                 help = 'hidden layer nodes for inside outside',
                                 type=int, default=80)
    parser_training.add_argument('-H2','--hidden2_insideoutside',
                                 help = 'hidden layer nodes2 for inside outside',
                                 type=int)
    parser_training.add_argument('-h1','--hidden_merge',
                                 help = 'hidden layer nodes for merge phase',
                                 type=int, default=80)
    parser_training.add_argument('-h2','--hidden2_merge',
                                 help = 'hidden layer nodes2 for merge phase',
                                 type=int)
    parser_training.add_argument('-f','--feature',
                                 help = 'feature length',
                                 type=int, default=80)

    parser_training.add_argument('-hn1','--hidden1',
                                 help = 'hidden layer nodes for neighbor model',
                                 type=int, default=200)
    parser_training.add_argument('-hn2','--hidden2',
                                 help = 'hidden layer nodes2 for neighbor model',
                                 type=int, default=50)
    parser_training.add_argument('-n','--neighbor',
                                 help = 'length of neighbor bases to see',
                                 type=int, default=40)

    # training option
    parser_training.add_argument('-g','--gamma',
                                 help = 'balance between the sensitivity and specificity ',
                                 type=float, action='append')
    parser_training.add_argument('-m','--positive-margin',
                                 help = 'margin for positives',
                                 type=float, default=0.2)
    parser_training.add_argument('--negative-margin',
                                 help = 'margin for negatives',
                                 type=float, default=0.2)
    parser_training.add_argument('-fu','--fully_learn',
                                 help = 'calculate loss for all canonical pair',
                                 action = 'store_true')
    parser_training.add_argument('-ip','--ipknot',
                                 help = 'predict pseudoknotted secaondary structure',
                                 action = 'store_true')

    parser_training.set_defaults(func = train)


    # add subparser for test
    parser_test = subparser.add_parser('test', help='test secondary structures')
    parser_test.add_argument('test_file',
                             help = 'FASTA or BPseq file for test',
                             type=argparse.FileType('r'))
    parser_test.add_argument('-p', '--parameters', help = 'Initial parameter file',
                             type=str, default="NEURALfold_parameters")
    parser_test.add_argument('-bp','--bpseq', help =
                             'use bpseq format',
                             action = 'store_true')
    parser_test.add_argument('-ip','--ipknot',
                             help = 'predict pseudoknotted secaondary structure',
                             action = 'store_true')
    parser_test.add_argument('-g','--gamma',
                             help = 'balance between the sensitivity and specificity ',
                             type=float, action='append')

    parser_test.set_defaults(func = test)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
