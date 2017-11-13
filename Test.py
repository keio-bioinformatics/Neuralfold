import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers


class Test:
    def __init__(self, seq_set, structure_set):
        self.seq_set = seq_set
        self.structure_set = structure_set
    def test(self):
        model =  Recursive.Recursive_net()
        serializers.load_npz("NEURALfold_params.data", model)
        predicted_structure_set = []
        for seq, true_structure in zip(self.seq_set, self.structure_set):
            inference = Inference.Inference(seq)
            predicted_BP = inference.ComputeInsideOutside(model)
            predicted_structure_set.append(inference.ComputePosterior(predicted_BP))
        return predicted_structure_set
