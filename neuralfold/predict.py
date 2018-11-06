import numpy as np
from . import Config
from . import Recursive
from . import Inference
from . import SStruct
from . import evaluate
from .decode.ipknot import IPknot
from .decode.nussinov import Nussinov
from .model.mlp import MLP
import pickle
from chainer import serializers

class Predict:
    def __init__(self, args):

        sstruct = SStruct.SStruct(args.seq_file)
        if args.bpseq:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_BPseq()
        else:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_FASTA()

        self.model = self.load_model(args.parameters)

        self.feature = 80
        self.ipknot = args.ipknot
        self.seq_file = args.seq_file
        if self.ipknot:
            self.decoder = IPknot()    
            if args.gamma:
                self.gamma = args.gamma
            else:
                self.gamma = [3.0,3.0]
        else:
            self.decoder = Nussinov()
            if args.gamma:
                self.gamma = args.gamma[0]
            else:
                self.gamma = 3.0
        self.args = args


    def run(self):
        predicted_structure_set = []

        for name, seq in zip(self.name_set, self.seq_set):
            print(name)
            predicted_BP = self.model.compute_bpp(seq)
            predicted_structure = self.decoder.decode(predicted_BP.data, gamma=self.gamma)

            print(seq)
            print(self.decoder.dot_parenthesis(seq, predicted_structure))
            predicted_structure_set.append(predicted_structure)

        if self.structure_set:
            Sensitivity, PPV, F_value = evaluate.get_score(self.structure_set, predicted_structure_set)
            return Sensitivity, PPV, F_value

        else:
            return 0, 0, 0


    def load_model(self, f):
        with open(f+'.pickle', 'rb') as fp:
            obj = pickle.load(fp)
            learning_model = obj[0]
            obj.pop(0)
            if learning_model == "recursive":
                model = Recursive.Recursive_net(*obj)

            elif learning_model == "deepnet":
                model = MLP(*obj)

            else:
                print("unexpected network")
                return None

        serializers.load_npz(f+'.npz', model)

        return model

    @classmethod
    def add_args(cls, parser):
        import argparse
        # add subparser for test
        parser_pred = parser.add_parser('predict', help='Predict RNA secondary structures for given sequences')
        parser_pred.add_argument('seq_file',
                                help = 'FASTA or BPseq file for prediction',
                                nargs='+',
                                type=argparse.FileType('r'))
                                # type=open)
        parser_pred.add_argument('-p', '--parameters', help = 'Initial parameter file',
                                type=str, default="NEURALfold_parameters")
        parser_pred.add_argument('-bp','--bpseq', help =
                                'use bpseq format',
                                action = 'store_true')
        parser_pred.add_argument('-ip','--ipknot',
                                help = 'predict pseudoknotted secondary structures',
                                action = 'store_true')
        parser_pred.add_argument('-g','--gamma',
                                help = 'balance between the sensitivity and specificity ',
                                type=float, action='append')

        parser_pred.set_defaults(func = lambda args: Predict(args).run())
