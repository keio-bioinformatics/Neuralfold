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
from .model.cnn import CNN
from .model.resnet import ResNet
from .model.lstm import LSTM2D
# from .model.wncnn import WNCNN
# from .model.wncnn2d import WNCNN2D
# from .model.mlp import MLP
# from .model.rnn import RNN
from .seq import load_seq, load_seq_from_list
from .structured_loss import StructuredLoss
from .piecewise_loss import PiecewiseLoss
from .labelwise_accuracy_loss import LabelwiseAccuracyLoss

class Train:
    def __init__(self, args):
        if args.seed >= 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            if chainer.backends.cuda.available and args.gpu >= 0:
                chainer.backends.cuda.cupy.random.seed(args.seed)
                chainer.global_config.cudnn_deterministic = True

        # load sequences
        self.name_set, self.seq_set, self.structure_set = load_seq_from_list(args.train_file)
        if args.test_file:
            self.test_files = [list(zip(*load_seq_from_list(f))) for f in args.test_file]
        else:
            self.test_files = []

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
            self.model = load_model(args.init_parameters, args)
        else:
            try:
                klass = globals()[args.learning_model]
                self.model = klass(**klass.parse_args(args))
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

        # loss function setup
        if True:
            self.net = LabelwiseAccuracyLoss(self.model, decoder, 
                                        compute_accuracy=self.compute_accuracy,
                                        verbose=args.verbose, 
                                        **LabelwiseAccuracyLoss.parse_args(args))
        elif False:
            self.net = StructuredLoss(self.model, decoder, 
                                    compute_accuracy=self.compute_accuracy,
                                    verbose=args.verbose, 
                                    **StructuredLoss.parse_args(args))
        else:
            self.net = PiecewiseLoss(self.model, decoder, 
                                    compute_accuracy=self.compute_accuracy,
                                    verbose=args.verbose, 
                                    **PiecewiseLoss.parse_args(args))
        if args.gpu >= 0:
            self.net.to_gpu(args.gpu)
        

        # optmizer setup
        if args.optimizer == 'Adam':
            self.optimizer = optimizers.Adam(alpha=args.adam_alpha, adabound=True)
        elif args.optimizer == 'MomentumSGD':
            self.optimizer = optimizers.MomentumSGD(lr=args.momentum_sgd_lr)
        else:
            raise 'unsupported optimizer'
        self.optimizer.setup(self.net)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))


    def run(self):
        converter = lambda batch, _: tuple(zip(*batch))
        training_data = list(zip(self.name_set, self.seq_set, self.structure_set))
        train_iter = iterators.SerialIterator(training_data, self.batchsize)
        # optimizer = optimizers.Adam(alpha=self.lr)
        # self.optimizer.setup(self.net)
        updater = training.StandardUpdater(train_iter, self.optimizer, converter=converter)
        trainer = training.Trainer(updater, (self.epochs, 'epoch'), out=self.outdir)
        if self.outdir:
            trainer.extend(extensions.LogReport())
            trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
            rep_name = ['epoch', 'elapsed_time', 'main/loss']
            if self.compute_accuracy:
                rep_name.append('main/f_val')
            plot_loss_name = ['main/loss']
            plot_f_val_name = ['main/f_val']
            for i, valid_data in enumerate(self.test_files):
                name = 'val' if i == 0 else 'val{}'.format(i)
                valid_iter = iterators.SerialIterator(valid_data, self.batchsize, False, False)
                trainer.extend(extensions.Evaluator(valid_iter, self.net, converter=converter), name=name)
                rep_name.append('{}/main/loss'.format(name))
                plot_loss_name.append('{}/main/loss'.format(name))
                if self.compute_accuracy:
                    rep_name.append('{}/main/f_val'.format(name))
                    plot_f_val_name.append('{}/main/f_val'.format(name))
            trainer.extend(extensions.PrintReport(rep_name))
            trainer.extend(extensions.PlotReport(plot_loss_name, file_name='loss.png'))
            if self.compute_accuracy:
                trainer.extend(extensions.PlotReport(plot_f_val_name, 
                                x_key='epoch', trigger=(1, 'epoch'), file_name='accuracy.png'))
            #trainer.extend(extensions.dump_graph('main/loss'))
            #trainer.extend(extensions.DumpGraph('main/loss'))


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
                                    help='enable verbose output',
                                    action='store_true')
        parser_training.add_argument('-i','--epochs', 
                                    help='the number of epochs (default: 1)',
                                    type=int, default=1)
        parser_training.add_argument('--batchsize', 
                                    help='batch size (default: 1)', 
                                    type=int, default=1)
        parser_training.add_argument('--compute-accuracy', 
                                    help='compute accuracy during training',
                                    action='store_true')
        parser_training.add_argument('--optimizer',
                                    help='choose optimizer [Adam (default) or MomentumSGD]',
                                    choices=('Adam', 'MomentumSGD'),
                                    type=str, default='Adam')
        parser_training.add_argument('--adam-alpha',
                                    help='alpha for Adam optimizer (default: 0.001)',
                                    type=float, default=0.001)
        parser_training.add_argument('--momentum-sgd-lr', 
                                    help='learning rate for MomentumSGD (default: 0.01)',
                                    type=float, default=0.01)
        parser_training.add_argument('--weight-decay',
                                    help='weight decoy for optimizer (default: 1e-10)',
                                    type=float, default=1e-10)
        StructuredLoss.add_args(parser_training)

        # neural networks architecture
        parser_training.add_argument('-l','--learning-model',
                                    #help='Select a learning model [MLP (default), RNN, CNN, WNCNN, WNCNN2D]',
                                    #choices=('MLP', 'RNN', 'CNN', 'WNCNN', 'WNCNN2D'),
                                    help='Select a learning model [CNN (default), ResNet, LSTM2D]',
                                    choices=('CNN', 'ResNet', 'LSTM2D'),
                                    type=str, default='CNN')
        CNN.add_args(parser_training)
        ResNet.add_args(parser_training)
        LSTM2D.add_args(parser_training)
        # MLP.add_args(parser_training)
        # RNN.add_args(parser_training)
        # WNCNN.add_args(parser_training)        
        # WNCNN2D.add_args(parser_training)        

        # decoder option
        parser_training.add_argument('-g','--gamma',
                                    help='balance between the sensitivity and specificity (default: 4.0)',
                                    type=float, action='append')
        parser_training.add_argument('-d', '--decode',
                                    help='Select a decoder for secondary structure prediction [nussinov (default), ipknot]',
                                    choices=('nussinov', 'ipknot'),
                                    type=str, default='nussinov')
        IPknot.add_args(parser_training)
        Nussinov.add_args(parser_training)

        parser_training.set_defaults(func = lambda args: Train(args).run())
