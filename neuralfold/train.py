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
from .seq import load_seq
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
        self.name_set, self.seq_set, self.structure_set = load_seq(args.train_file)
        if args.test_file:
            self.name_set_test, self.seq_set_test, self.structure_set_test = load_seq(args.test_file)
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
        pos_margin = args.positive_margin
        neg_margin = args.negative_margin
        if args.ipknot:
            gamma = args.gamma if args.gamma is not None else (3.0, 3.0)
            decoder = IPknot(gamma)
        else:
            gamma = args.gamma if args.gamma is not None else 3.0
            decoder = Nussinov(gamma)

        # loss function
        self.net = StructuredLoss(self.model, decoder, pos_margin, neg_margin, 
                                compute_accuracy=self.compute_accuracy)


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
            trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
            trainer.extend(extensions.PlotReport(['main/f_val', 'val/main/f_val'], x_key='epoch', file_name='accuracy.png'))
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
        parser_training.add_argument('--gpu', help='set GPU ID (-1 for CPU)',
                                    type=int, default=-1)
        parser_training.add_argument('--seed', help='set the random seed for reproducibility',
                                    type=int, default=-1)

        # data
        parser_training.add_argument('train_file', help='FASTA or BPseq file for training', nargs='+',
                                    type=argparse.FileType('r'))
        parser_training.add_argument('-t', '--test_file', help='FASTA or BPseq file for test', nargs='+',
                                    type=argparse.FileType('r'))

        # training files
        parser_training.add_argument('--init-parameters', help='Initial parameter file',
                                    type=str)
        parser_training.add_argument('-p','--parameters', help='Output parameter file',
                                    type=str, default="NEURALfold_parameters")
        parser_training.add_argument('-o', '--trainer-output', 
                                    help='Output directory during training',
                                    type=str, default='')
        parser_training.add_argument('--resume', help='resume from snapshot',
                                    type=str, default='')

        # training parameters
        parser_training.add_argument('-i','--epochs', help='the number of epochs',
                                    type=int, default=1)
        parser_training.add_argument('--batchsize', help='batch size', 
                                    type=int, default=1)
        parser_training.add_argument('--compute_accuracy', help='compute accuracy during training',
                                    action='store_true')

        # neural networks architecture
        parser_training.add_argument('-l','--learning_model',
                                    help='learning_model',
                                    choices=('MLP', 'RNN'),
                                    type=str, default='MLP')
        MLP.add_args(parser_training)
        RNN.add_args(parser_training)

        # decoder option
        parser_training.add_argument('-g','--gamma',
                                    help='balance between the sensitivity and specificity ',
                                    type=float, action='append')
        parser_training.add_argument('-m','--positive-margin',
                                    help='margin for positives',
                                    type=float, default=0.2)
        parser_training.add_argument('--negative-margin',
                                    help='margin for negatives',
                                    type=float, default=0.2)
        parser_training.add_argument('-ip','--ipknot',
                                    help='predict pseudoknotted secaondary structure',
                                    action='store_true')

        parser_training.set_defaults(func = lambda args: Train(args).run())
