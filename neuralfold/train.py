import math
import os
import pickle
import random
from time import time

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import (Chain, Variable, cuda, iterators, optimizer, optimizers,
                     serializers, training)
from chainer.dataset import concat_examples
from chainer.training import extensions
import chainer.backends.cuda

from . import evaluate
from .decode.ipknot import IPknot
from .decode.nussinov import Nussinov
from .model import load_model
from .model.mlp import MLP
from .model.rnn import RNN
from .seq import load_seq, load_seq_from_list
from .structured_loss import StructuredLoss


class Train:
    def __init__(self, args):
        if args.seed >= 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            if chainer.backends.cuda.available:
                chainer.backends.cuda.cupy.random.seed(args.seed)
                chainer.global_config.cudnn_deterministic = True

        # load sequences
        self.name_set, self.seq_set, self.structure_set = load_seq_from_list(args.train_file)
        if args.test_file:
            self.name_set_test, self.seq_set_test, self.structure_set_test = load_seq_from_list(args.test_file)
        else:
            self.seq_set_test = None

        # output
        self.param_file = args.parameters        
        self.outdir = args.trainer_output

        # training parameters
        self.epochs = args.epochs
        self.batchsize = args.batchsize
        self.resume = args.resume
        self.compute_accuracy = args.compute_accuracy

        # model setup
        if args.init_parameters:
            self.model = load_model(args.init_parameters)
        else:
            try:
                klass = globals()[args.learning_model]
                self.model = klass(**klass.parse_args(args))
                if args.gpu >= 0:
                    self.model.to_gpu(args.gpu)
            except KeyError:
                raise RuntimeError("{} is unknown model class.".format(args.learning_model))

        # decoder setup
        if args.decode == 'ipknot':
            gamma = args.gamma if args.gamma is not None else (4.0, 2.0)
            decoder = IPknot(gamma, **IPknot.parse_args(args))
        elif args.decode == 'nussinov':
            gamma = args.gamma[-1] if args.gamma is not None else 4.0
            decoder = Nussinov(gamma, **Nussinov.parse_args(args))
        else:
            raise RuntimeError("Unknown decoder: {}".format(args.decode))

        # loss function
        self.net = StructuredLoss(self.model, decoder, 
                                compute_accuracy=self.compute_accuracy,
                                verbose=args.verbose, 
                                **StructuredLoss.parse_args(args))


    def run(self):
        converter = lambda batch, _: tuple(zip(*batch))
        training_data = list(zip(self.name_set, self.seq_set, self.structure_set))
        train_iter = iterators.SerialIterator(training_data, self.batchsize)
        optimizer = optimizers.Adam()
        optimizer.setup(self.net)
        updater = training.StandardUpdater(train_iter, optimizer, converter=converter)
        trainer = training.Trainer(updater, (self.epochs, 'epoch'), out=self.outdir)
        if self.outdir:
            trainer.extend(extensions.LogReport())
            trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
            rep_name = ['epoch', 'elapsed_time', 'main/loss']
            if self.compute_accuracy:
                rep_name.append('main/f_val')
            if self.seq_set_test is not None:
                valid_data = list(zip(self.name_set_test, self.seq_set_test, self.structure_set_test))
                valid_iter = iterators.SerialIterator(valid_data, self.batchsize, False, False)
                trainer.extend(extensions.Evaluator(valid_iter, self.net, converter=converter), name='val')
                rep_name.append('val/main/loss')
                if self.compute_accuracy:
                    rep_name.append('val/main/f_val')
            trainer.extend(extensions.PrintReport(rep_name))
            trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], file_name='loss.png'))
            if self.compute_accuracy:
                trainer.extend(extensions.PlotReport(['main/f_val', 'val/main/f_val'], 
                                x_key='epoch', trigger=(1, 'epoch'), file_name='accuracy.png'))
            #trainer.extend(extensions.dump_graph('main/loss'))


        if self.resume:
            serializers.load_npz(self.resume, trainer)
        trainer.run()

        self.model.save_model(self.param_file)

        return


    @classmethod
    def add_args(cls, parser):
        import argparse
        parser_training = parser.add_parser('train', help='training RNA secondary structures')
        parser_training.add_argument('--gpu', 
                                    help='set GPU ID (-1 for CPU)',
                                    type=int, default=-1)
        parser_training.add_argument('--seed', 
                                    help='set the random seed for reproducibility',
                                    type=int, default=-1)

        # data
        parser_training.add_argument('train_file', 
                                    help='The list of FASTA or BPseq files for training', nargs='+',
                                    type=argparse.FileType('r'))
        parser_training.add_argument('-t', '--test-file', 
                                    help='The list of FASTA or BPseq files for test', nargs='+',
                                    type=argparse.FileType('r'))

        # training files
        parser_training.add_argument('--init-parameters', 
                                    help='Initial parameter file',
                                    type=str)
        parser_training.add_argument('-p','--parameters', 
                                    help='Output parameter file',
                                    type=str, default="NEURALfold_parameters")
        parser_training.add_argument('-o', '--trainer-output', 
                                    help='Output directory during training',
                                    type=str, default='')
        parser_training.add_argument('--resume', help='resume from snapshot',
                                    type=str, default='')

        # training parameters
        parser_training.add_argument('-v', '--verbose', 
                                    help='verbose output',
                                    action='store_true')
        parser_training.add_argument('-i','--epochs', 
                                    help='the number of epochs',
                                    type=int, default=1)
        parser_training.add_argument('--batchsize', 
                                    help='batch size', 
                                    type=int, default=1)
        parser_training.add_argument('--compute-accuracy', 
                                    help='compute accuracy during training',
                                    action='store_true')
        StructuredLoss.add_args(parser_training)

        # neural networks architecture
        parser_training.add_argument('-l','--learning-model',
                                    help='Select a learning model',
                                    choices=('MLP', 'RNN'),
                                    type=str, default='MLP')
        MLP.add_args(parser_training)
        RNN.add_args(parser_training)

        # decoder option
        parser_training.add_argument('-g','--gamma',
                                    help='balance between the sensitivity and specificity',
                                    type=float, action='append')
        parser_training.add_argument('-d', '--decode',
                                    help='Select a decoder for secondary structure prediction',
                                    choices=('nussinov', 'ipknot'),
                                    type=str, default='nussinov')
        IPknot.add_args(parser_training)
        Nussinov.add_args(parser_training)

        parser_training.set_defaults(func = lambda args: Train(args).run())
