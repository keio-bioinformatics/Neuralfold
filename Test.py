import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
import SStruct
import Deepnet
import Evaluate
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
        self.gamma = args.gamma
        if self.ipknot:
            self.gamma = (self.gamma, self.gamma)
        self.args = args


    def test(self):
        predicted_structure_set = []

        for name,seq in zip(self.name_set, self.seq_set):
            print(name)

            inference = Inference.Inference(seq, self.feature)
            predicted_BP = inference.ComputeNeighbor(self.model)
            predicted_structure = inference.ComputePosterior(predicted_BP.data, self.ipknot, self.gamma)

            print(inference.seq)
            print(inference.dot_parentheis(predicted_structure))

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
