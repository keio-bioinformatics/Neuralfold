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

from . import evaluate
from .decode.ipknot import IPknot
from .decode.nussinov import Nussinov
from .model import load_model
from .model.mlp import MLP
from .model.rnn import RNN
from .seq import load_seq


class Train:
    def __init__(self, args):
        # load sequences
        self.name_set, self.seq_set, self.structure_set = load_seq(args.train_file)
        if args.test_file:
            self.name_set_test, self.seq_set_test, self.structure_set_test = load_seq(args.test_file)
        else:
            self.seq_set_test = None

        # output
        self.param_file = args.parameters        

        # model setup
        if args.init_parameters:
            self.model = load_model(args.init_parameters)
        else:
            try:
                klass = globals()[args.learning_model]
                self.model = klass(**klass.parse_args(args))
            except KeyError:
                raise RuntimeError("{} is unknown model class.".format(args.learning_model))

        # optimizer setup
        self.epochs = args.epochs
        self.batchsize = args.batchsize
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        if args.init_optimizers:
            serializers.load_npz(args.init_optimizers, self.optimizer)

        # decoder setup
        self.pos_margin = args.positive_margin
        self.neg_margin = args.negative_margin
        if args.ipknot:
            gamma = args.gamma if args.gamma is not None else (3.0, 3.0)
            self.decoder = IPknot(gamma)
        else:
            gamma = args.gamma if args.gamma is not None else 3.0
            self.decoder = Nussinov(gamma)


    def calculate_loss(self, name_set, seq_set, true_structure_set):
        loss = 0
        print(name_set)
        predicted_BP = self.model.compute_bpp(seq_set)
        for k, (name, seq, true_structure) in enumerate(zip(name_set, seq_set, true_structure_set)):
            N = len(seq)
            print(name, N, 'bp')
            margin = np.full((N, N), self.neg_margin, dtype=np.float32)
            for i, j in true_structure:
                margin[i, j] -= self.pos_margin + self.neg_margin

            predicted_structure = self.decoder.decode(predicted_BP[k].array, margin=margin)
            predicted_score = self.decoder.calc_score(predicted_BP[k], pair=predicted_structure, margin=margin)
            true_score = self.decoder.calc_score(predicted_BP[k], pair=true_structure)
            loss += predicted_score - true_score

        return loss


    def run(self):
        training_data = list(zip(self.name_set, self.seq_set, self.structure_set))
        train_iter = iterators.SerialIterator(training_data, self.batchsize)

        updater = training.StandardUpdater(train_iter, self.optimizer, 
            converter=lambda batch, device: tuple(zip(*batch)),
            loss_func=self.calculate_loss)
        trainer = training.Trainer(updater, (self.epochs, 'epoch'), out='res')
        trainer.extend(extensions.LogReport())
        #trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
        trainer.run()

        self.model.save_model(self.param_file)

        return

        # while train_iter.epoch < self.epochs:
        #     print('epoch =', train_iter.epoch)
        #     train_batch = train_iter.next()
        #     name, seq, true_structure = zip(*train_batch)
        #     loss = self.calculate_loss(name, seq, true_structure)

        #     self.model.zerograds()
        #     loss.backward()
        #     self.optimizer.update()

        #     if train_iter.is_new_epoch:
        #         # save model
        #         self.model.save_model(self.param_file + str(train_iter.epoch))
        #         serializers.save_npz(self.param_file + "_state" + str(train_iter.epoch) + ".npz", self.optimizer)

            #TEST
            # if (self.test_file is not None):
            #     predicted_structure_set = []
            #     print("start testing...")
            #     for seq in self.seq_set_test:
            #         inference = Inference.Inference(seq,self.feature)
            #         if self.args.learning_model == "recursive":
            #             predicted_BP = inference.ComputeInsideOutside(self.model)
            #         elif self.args.learning_model == "deepnet":
            #             predicted_BP = inference.ComputeNeighbor(self.model)
            #         else:
            #             print("unexpected network")

            #         predicted_structure = inference.ComputePosterior(predicted_BP.data, self.ipknot, self.gamma)
            #         if len(predicted_structure) == 0:
            #             continue
            #         if self.ipknot:
            #             predicted_structure = predicted_structure[0] + predicted_structure[1]
            #         predicted_structure_set.append(predicted_structure)

            #     # print(predicted_structure_set[0])
            #     # print(self.structure_set_test[0])

            #     evaluate = Evaluate.Evaluate(predicted_structure_set, self.structure_set_test)
            #     Sensitivity, PPV, F_value = evaluate.get_score()

            #     file = open('result.txt', 'a')
            #     result = ['Sensitivity=', str(round(Sensitivity,5)),' ',
            #               'PPV=', str(round(PPV,5)),' ',
            #               'F_value=', str(round(F_value,5)),' ',str(self.gamma),'\n']
            #     file.writelines(result)
            #     file.close()
            #     print(Sensitivity, PPV, F_value)

            # print('ite'+str(ite)+': it cost '+str(time() - start)+'sec')

        # self.model.save_model(self.param_file)

        # return self.model


    @classmethod
    def add_args(cls, parser):
        import argparse
        parser_training = parser.add_parser('train', help='training RNA secondary structures')

        # data
        parser_training.add_argument('train_file', help = 'FASTA or BPseq file for training', nargs='+',
                                    type=argparse.FileType('r'))
        parser_training.add_argument('-t','--test_file', help = 'FASTA or BPseq file for test', nargs='+',
                                    type=argparse.FileType('r'))
        parser_training.add_argument('-bp','--bpseq', help = 'Use bpseq',
                                    action = 'store_true')

        # training files
        parser_training.add_argument('--init-parameters', help = 'Initial parameter file',
                                    type=str)
        parser_training.add_argument('-p','--parameters', help = 'Output parameter file',
                                    type=str, default="NEURALfold_parameters")
        parser_training.add_argument('--init-optimizers', help = 'Initial optimizer state file',
                                    type=str)
        parser_training.add_argument('-o','--optimizers', help = 'Optimizer state file',
                                    type=str, default="state.npz")
        parser_training.add_argument('-i','--epochs', help = 'the number of epochs',
                                    type=int, default=1)
        parser_training.add_argument('--batchsize', help='batch size', 
                                    type=int, default=1)

        # neural networks architecture
        parser_training.add_argument('-l','--learning_model',
                                    help = 'learning_model',
                                    choices=('MLP', 'RNN'),
                                    type=str, default='MLP')
        MLP.add_args(parser_training)
        RNN.add_args(parser_training)

        # decoder option
        parser_training.add_argument('-g','--gamma',
                                    help = 'balance between the sensitivity and specificity ',
                                    type=float, action='append')
        parser_training.add_argument('-m','--positive-margin',
                                    help = 'margin for positives',
                                    type=float, default=0.2)
        parser_training.add_argument('--negative-margin',
                                    help = 'margin for negatives',
                                    type=float, default=0.2)
        # parser_training.add_argument('-fu','--fully_learn',
        #                             help = 'calculate loss for all canonical pair',
        #                             action = 'store_true')
        parser_training.add_argument('-ip','--ipknot',
                                    help = 'predict pseudoknotted secaondary structure',
                                    action = 'store_true')

        parser_training.set_defaults(func = lambda args: Train(args).run())
