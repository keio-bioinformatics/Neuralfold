import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
import SStruct
import Deepnet
import Evaluate
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
# from multiprocessing import Pool
# import multiprocessing as multi
maximum_slots = Config.maximum_slots
batch_num = Config.batch_num

class Test:
    def __init__(self, args):

        sstruct = SStruct.SStruct(args.test_file, False)
        if args.bpseq:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_BPseq()
        else:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_FASTA()

        self.model = Deepnet.Deepnet(200, 50, False, "sigmoid")

        # if args.Parameters:
        #     serializers.load_npz(args.Parameters.name, self.model)
        # else:
        serializers.load_npz("NEURALfold_params.data", self.model)

        self.neighbor = 40
        self.feature = 80
        self.activation_function = "sigmoid"
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

            inference = Inference.Inference(seq,self.feature, self.activation_function)
            predicted_BP, predicted_UP_left, predicted_UP_right = inference.ComputeNeighbor(self.model, neighbor=self.neighbor)
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
