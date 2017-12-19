import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
import SStruct
import Evaluate
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
from multiprocessing import Pool
import multiprocessing as multi
maximum_slots = Config.maximum_slots
batch_num = Config.batch_num

class Test:
    def __init__(self, args):
        sstruct = SStruct.SStruct(args.test_file)
        self.name_set_test, self.seq_set_test, self.structure_set_test = sstruct.load_FASTA()
        self.model =  Recursive.Recursive_net()
        if args.Parameters:
            serializers.load_npz(args.Parameters.name, self.model)
        self.args = args

    def test_Parallel(self,seq):
        inference = Inference.Inference(seq,80)
        predicted_BP = inference.ComputeInsideOutside(self.model)
        # print(predicted_BP)
        return inference.ComputePosterior(predicted_BP)

    def test(self):
        # p= Pool(maximum_slots)
        # p= Pool(batch_num)
        # p= Pool(2)
        # p= Pool(multi.cpu_count())
        # predicted_structure_set = p.map(self.test_Parallel, self.seq_set_test)
        # p.close()
        predicted_structure_set = []
        for seq in self.seq_set_test:
            inference = Inference.Inference(seq,80)
            predicted_BP = inference.ComputeInsideOutside(self.model)
            predicted_structure_set.append(inference.ComputePosterior(predicted_BP))
        evaluate = Evaluate.Evaluate(predicted_structure_set , self.structure_set_test)
        Sensitivity, PPV, F_value = evaluate.getscore()
        return Sensitivity, PPV, F_value
