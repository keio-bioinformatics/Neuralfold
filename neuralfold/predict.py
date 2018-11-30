import pickle

import chainer
import numpy as np
from chainer import serializers

from . import evaluate
from .decode.ipknot import IPknot
from .decode.nussinov import Nussinov
from .model import load_model
from .model.mlp import MLP
from .seq import load_seq


class Predict:
    def __init__(self, args):
        self.name_set, self.seq_set, self.structure_set = load_seq(args.seq_file)
        self.model = load_model(args.parameters)
        if args.gpu >= 0:
            self.model.to_gpu(args.gpu)

        if args.decode == 'ipknot':
            gamma = args.gamma if args.gamma is not None else (4.0, 2.0)
            self.decoder = IPknot(gamma, **IPknot.parse_args(args))
        elif args.decode == 'nussinov':
            gamma = args.gamma[-1] if args.gamma is not None else 4.0
            self.decoder = Nussinov(gamma, **Nussinov.parse_args(args))
        else:
            raise RuntimeError("Unknown decoder: {}".format(args.decode))


    def run(self):
        predicted_structure_set = []

        for name, seq in zip(self.name_set, self.seq_set):
            N = len(seq)
            print(name)
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                predicted_BP = self.model.compute_bpp([seq])
            predicted_structure = self.decoder.decode(seq, predicted_BP[0].array[0:N, 0:N])

            print(seq)
            print(self.decoder.dot_parenthesis(seq, predicted_structure))
            predicted_structure_set.append(predicted_structure)

        if self.structure_set:
            Sensitivity, PPV, F_value = evaluate.get_score(self.structure_set, predicted_structure_set)
            return Sensitivity, PPV, F_value

        else:
            return 0, 0, 0


    @classmethod
    def add_args(cls, parser):
        import argparse
        # add subparser for test
        parser_pred = parser.add_parser('predict', help='Predict RNA secondary structures for given sequences')
        parser_pred.add_argument('--gpu', help='set GPU ID (-1 for CPU)',
                                type=int, default=-1)

        parser_pred.add_argument('seq_file',
                                help='FASTA or BPseq file for prediction',
                                nargs='+',
                                type=argparse.FileType('r'))
        parser_pred.add_argument('-p', '--parameters', help = 'Initial parameter file',
                                type=str, default="NEURALfold_parameters")
        # parser_pred.add_argument('-bp','--bpseq', help='Use bpseq format',
        #                         action='store_true')
        parser_pred.add_argument('-g','--gamma',
                                help='Balance between sensitivity and specificity ',
                                type=float, action='append')
        parser_pred.add_argument('-d', '--decode',
                                help='Select a decoder for secondary structure prediction (default: nussinov)',
                                choices=('nussinov', 'ipknot'),
                                type=str, default='nussinov')
        Nussinov.add_args(parser_pred)
        IPknot.add_args(parser_pred)

        parser_pred.set_defaults(func = lambda args: Predict(args).run())
