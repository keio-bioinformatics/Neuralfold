import numpy as np
import sys
import SStruct
import Inference
import Evaluate
import Train
import Test
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

args = sys.argv
filename = args[2]

sstruct = SStruct.SStruct(filename)
name_set,seq_set,structure_set = sstruct.load_FASTA()
#print(name_set, seq_set, structure_set)

if args[1] == 'train':
    print("start training...")
    train = Train.Train(seq_set, structure_set)
    model = train.train()
    serializers.save_npz("NEURALfold_params.data", model)

elif args[1] == 'test':
    print("start testing...")
    test = Test.Test(seq_set, structure_set)
    predicted_structure_set = test.test()
    evaluate = Evaluate.Evaluate(predicted_structure_set , structure_set)
    Sensitivity, PPV, F_value = evaluate.getscore()
    print(Sensitivity, PPV, F_value)
else:
    print("input train or test")
#else:
