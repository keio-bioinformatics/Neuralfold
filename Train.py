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

        for seq, true_structure in zip(self.seq_set, self.structure_set):
            inference = Inference.Inference(seq)
            predicted_BP = inference.ComputeInsideOutside(model)
            predicted_structure = inference.ComputePosterior(predicted_BP)
            match_pairs = []
            for predicted_pair in predicted_structure:
                for true_pair in true_structure:
                    if predicted_pair == true_pair:
                        match_pairs.append(predicted_pair)

            for predicted_pair in predicted_structure:
                for match_pair in match_pairs:
                    if predicted_pair == match_pair:

            for true_pair in true_structure:
                for match_pair in match_pairs:
