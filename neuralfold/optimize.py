import math
import os
import pickle
import random
from time import time

import chainer.backends.cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
import optuna
from chainer import (Chain, Variable, cuda, iterators, optimizer, optimizers,
                     serializers, training)
from chainer.dataset import concat_examples
from chainer.training import extensions

from . import evaluate
from .decode.ipknot import IPknot
from .decode.nussinov import Nussinov
from .model import load_model
from .model.cnn import CNN
from .model.wncnn import WNCNN
from .model.mlp import MLP
from .model.rnn import RNN
from .piecewise_loss import PiecewiseLoss
from .seq import load_seq, load_seq_from_list
from .structured_loss import StructuredLoss


class Optimize:
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
        self.outdir = args.trainer_output

        # optuna
        self.study_name = args.study_name
        self.storage = args.storage
        self.trials = args.trials

        # training parameters
        self.epochs = args.epochs
        self.compute_accuracy = True
        self.gpu = args.gpu
        self.batchsize = args.batchsize

        # decoder setup
        if args.decode == 'ipknot':
            gamma = args.gamma if args.gamma is not None else (4.0, 2.0)
            self.decoder = IPknot(gamma, **IPknot.parse_args(args))
        elif args.decode == 'nussinov':
            gamma = args.gamma[-1] if args.gamma is not None else 4.0
            self.decoder = Nussinov(gamma, **Nussinov.parse_args(args))
        else:
            raise RuntimeError("Unknown decoder: {}".format(args.decode))

    # for CNN
    # def create_model(self, trial):
    #     cnn_use_dilate = True #trial.suggest_categorical('cnn_use_dilate', [True, False])
    #     if cnn_use_dilate:
    #         cnn_width = 1
    #     else:
    #         cnn_width = trial.suggest_int('cnn_width', 1, 4)
    #     cnn_layers = trial.suggest_int('cnn_layers', 1, 6)
    #     cnn_channels = int(trial.suggest_loguniform('cnn_channels', 32//2, 256//2))
    #     cnn_hidden_nodes = int(trial.suggest_loguniform('cnn_hidden_nodes', 32, 256))
    #     cnn_use_bn = True #trial.suggest_categorical('cnn_use_bn', [True, False])
    #     cnn_use_dropout = trial.suggest_categorical('cnn_use_dropout', [True, False])
    #     if cnn_use_dropout:
    #         cnn_dropout_rate = trial.suggest_uniform('cnn_dropout_rate', 0.0, 0.25)
    #     else:
    #         cnn_dropout_rate = None
    #     pos_margin = trial.suggest_uniform('pos_margin', 0.0, 0.5)
    #     neg_margin = trial.suggest_uniform('neg_margin', 0.0, pos_margin)

    #     model = CNN(layers=cnn_layers, channels=cnn_channels, 
    #                 width=cnn_width, hidden_nodes=cnn_hidden_nodes,
    #                 dropout_rate=cnn_dropout_rate, 
    #                 use_dilate=cnn_use_dilate, use_bn=cnn_use_bn)

    #     net = StructuredLoss(model, self.decoder, 
    #                             compute_accuracy=self.compute_accuracy,
    #                             positive_margin=pos_margin, negative_margin=neg_margin)

    #     return net

    # for Weight Normalizaion CNN
    def create_model(self, trial):
        wncnn_layers = trial.suggest_int('wncnn_layers', 1, 6)
        wncnn_channels = int(trial.suggest_loguniform('wncnn_channels', 32//2, 256//2))
        wncnn_hidden_nodes = int(trial.suggest_loguniform('wncnn_hidden_nodes', 32, 256))
        wncnn_dropout_rate = trial.suggest_uniform('wncnn_dropout_rate', 0.0, 0.25)
        pos_margin = trial.suggest_uniform('pos_margin', 0.0, 0.5)
        neg_margin = trial.suggest_uniform('neg_margin', 0.0, pos_margin)

        model = WNCNN(layers=wncnn_layers, channels=wncnn_channels, 
                    width=1, hidden_nodes=wncnn_hidden_nodes,
                    dropout_rate=wncnn_dropout_rate)

        net = StructuredLoss(model, self.decoder, 
                                compute_accuracy=self.compute_accuracy,
                                positive_margin=pos_margin, negative_margin=neg_margin)

        return net
    

    def create_optimizer(self, trial, model):
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
        if optimizer_name == 'Adam':
            adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-3, 1e-1)
            optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
        else:
            momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-3, 1e-1)
            optimizer = chainer.optimizers.MomentumSGD(lr=momentum_sgd_lr)

        weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        return optimizer


    def objective(self, trial):
        training_data = list(zip(self.name_set, self.seq_set, self.structure_set))
        batchsize = self.batchsize if self.batchsize is not None else trial.suggest_int('batch_size', 1, 16)
        train_iter = iterators.SerialIterator(training_data, batchsize)
        model = self.create_model(trial)
        optimizer = self.create_optimizer(trial, model)
        converter = lambda batch, _: tuple(zip(*batch))
        updater = training.StandardUpdater(train_iter, optimizer, converter=converter)
        trainer = training.Trainer(updater, (self.epochs, 'epoch'), out=self.outdir)

        log_report_extension = chainer.training.extensions.LogReport(log_name=None)
        trainer.extend(log_report_extension)
        trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
        rep_name = ['epoch', 'elapsed_time', 'main/loss']
        if self.compute_accuracy:
            rep_name.append('main/f_val')
        if self.seq_set_test is not None:
            valid_data = list(zip(self.name_set_test, self.seq_set_test, self.structure_set_test))
            valid_iter = iterators.SerialIterator(valid_data, batchsize, False, False)
            trainer.extend(extensions.Evaluator(valid_iter, model, converter=converter), name='val')
            rep_name.append('val/main/loss')
            if self.compute_accuracy:
                rep_name.append('val/main/f_val')
        trainer.extend(extensions.PrintReport(rep_name))
        trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], file_name='loss.png'))
        if self.compute_accuracy:
            trainer.extend(extensions.PlotReport(['main/f_val', 'val/main/f_val'], 
                            x_key='epoch', trigger=(1, 'epoch'), file_name='accuracy.png'))
        #trainer.extend(extensions.dump_graph('main/loss'))

        trainer.run()

        log_last = log_report_extension.log[-1]
        for key, value in log_last.items():
            trial.set_user_attr(key, value)

        val_f_val = 1.0 - log_report_extension.log[-1]['val/main/f_val']

        return val_f_val

    def run(self):
        if self.study_name is not None and self.storage is not None:
            study = optuna.Study(study_name=self.study_name, storage=self.storage)
        else:
            study = optuna.create_study()
        study.optimize(self.objective, n_trials=self.trials)

        print('Number of finished trials: ', len(study.trials))

        print('Best trial:')
        trial = study.best_trial

        print('  Value: ', trial.value)

        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        print('  User attrs:')
        for key, value in trial.user_attrs.items():
            print('    {}: {}'.format(key, value))


    @classmethod
    def add_args(cls, parser):
        import argparse
        parser_training = parser.add_parser('optimize', help='optimize hyper-parameters')

        parser_training.add_argument('--trials',
                                    help='The number of trials for hyperparameter optimization',
                                    type=int, default=100)
        parser_training.add_argument('--study-name',
                                    help='Set optuna study name',
                                    type=str)
        parser_training.add_argument('--storage',
                                    help='Set optuna storage',
                                    type=str)

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
        parser_training.add_argument('-o', '--trainer-output', 
                                    help='Output directory during training',
                                    type=str, default='out')

        # training parameters
        parser_training.add_argument('-v', '--verbose', 
                                    help='verbose output',
                                    action='store_true')
        parser_training.add_argument('-i','--epochs', 
                                    help='the number of epochs',
                                    type=int, default=1)
        parser_training.add_argument('--batchsize', 
                                    help='batch size', 
                                    type=int)
        # parser_training.add_argument('--compute-accuracy', 
        #                             help='compute accuracy during training',
        #                             action='store_true')
        # StructuredLoss.add_args(parser_training)

        # neural networks architecture
        # parser_training.add_argument('-l','--learning-model',
        #                             help='Select a learning model',
        #                             choices=('MLP', 'RNN', 'CNN'),
        #                             type=str, default='MLP')
        # MLP.add_args(parser_training)
        # RNN.add_args(parser_training)
        # CNN.add_args(parser_training)

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

        parser_training.set_defaults(func = lambda args: Optimize(args).run())
