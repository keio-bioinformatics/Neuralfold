import numpy as np
from . import Config
from . import Recursive
import chainer.links as L
import chainer.functions as F
from . import Inference
from . import SStruct
from . import Deepnet
from . import Evaluate
import pickle
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

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
            if args.gamma:
                self.gamma = args.gamma
            else:
                self.gamma = [3.0,3.0]
        else:
            if args.gamma:
                self.gamma = args.gamma[0]
            else:
                self.gamma = 3.0
        self.args = args


    def test(self):
        predicted_structure_set = []

        for name, seq, true_structure in zip(self.name_set, self.seq_set, self.structure_set):
            print(name)

            inference = Inference.Inference(seq, self.feature)
            predicted_BP = inference.ComputeNeighbor(self.model)
            predicted_structure = inference.ComputePosterior(predicted_BP.data, self.ipknot, self.gamma)

            print(inference.seq)
            print(inference.dot_parentheis(predicted_structure))
            print(inference.dot_parentheis(true_structure))

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
                model = Deepnet.Deepnet(*obj)

            else:
                print("unexpected network")
                return None

        serializers.load_npz(f+'.npz', model)

        return model
