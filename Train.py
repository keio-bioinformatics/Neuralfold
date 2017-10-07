import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers


class Train:
    def __init__(self, seq_set, structure_set):
        self.seq_set = seq_set
        self.structure_set = structure_set

    def train():
        #define model
        model = Recursive.Recursive_net()
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        for seq, structure in zip(self.seq_set, self.structure_set):
            inference = Inference.Inference(s)
            predicted_BP = inference.ComputeInsideOutside(model)
