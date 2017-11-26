import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
from multiprocessing import Pool
import multiprocessing as multi

class Test:
    def __init__(self, seq_set_test):
        self.seq_set_test = seq_set_test

    def test_Parallel(self,seq):
        inference = Inference.Inference(seq)
        predicted_BP = inference.ComputeInsideOutside(self.model)
        return inference.ComputePosterior(predicted_BP)

    def test(self):
        self.model =  Recursive.Recursive_net()
        serializers.load_npz("NEURALfold_params.data", self.model)
        # p= Pool(80)
        p= Pool(multi.cpu_count())
        predicted_structure_set = p.map(self.test_Parallel, self.seq_set_test)
        p.close()
        return predicted_structure_set
