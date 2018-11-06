import numpy as np
from . import Config
from . import Recursive
from . import Inference
from . import SStruct
from . import Evaluate
from .decode.ipknot import IPknot
from .decode.nussinov import Nussinov
from .model.mlp import MLP
import pickle
from chainer import serializers

class Test:
    def __init__(self, args):

        sstruct = SStruct.SStruct(args.test_file)
        if args.bpseq:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_BPseq()
        else:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_FASTA()

        self.model = self.load_model(args.parameters)

        self.feature = 80
        self.ipknot = args.ipknot
        self.test_file = args.test_file
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


    def test(self):
        predicted_structure_set = []

        for name, seq, true_structure in zip(self.name_set, self.seq_set, self.structure_set):
            print(name)

            #inference = Inference.Inference(seq, self.feature)
            #predicted_BP = inference.ComputeNeighbor(self.model)
            #predicted_structure = inference.ComputePosterior(predicted_BP.data, self.ipknot, self.gamma)
            predicted_BP = self.model.compute_bpp(seq)
            predicted_structure = self.decoder.decode(predicted_BP.data, gamma=self.gamma)

            print(seq)
            print(self.decoder.dot_parenthesis(seq, predicted_structure))
            print(self.decoder.dot_parenthesis(seq, true_structure))

            predicted_structure_set.append(predicted_structure)

        if self.structure_set:
            evaluate = Evaluate.Evaluate(predicted_structure_set , self.structure_set)
            Sensitivity, PPV, F_value = evaluate.getscore()
            return Sensitivity, PPV, F_value

        else:
            return 0,0,0


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
        parser_test = parser.add_parser('test', help='test secondary structures')
        parser_test.add_argument('test_file',
                                help = 'FASTA or BPseq file for test',
                                nargs='+',
                                type=argparse.FileType('r'))
                                # type=open)
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

        parser_test.set_defaults(func = lambda args: Test(args).test())
