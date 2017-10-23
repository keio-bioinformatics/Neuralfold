import numpy as np
import sys
import SStruct
#import Inference
#import Evaluate
import Train
args = sys.argv
filename = args[2]

sstruct = SStruct.SStruct(filename)
name_set,seq_set,structure_set = sstruct.load_FASTA()
print(name_set, seq_set, structure_set)

if args[1]=='train':
    train = Train.Train(seq_set, structure_set)
    train.train()


#elif args[1]=='test':
#    evaluate = Evaluate.Evaluate(predicted_structures , true_structures)
#    Accuracy,PPV,F_value = evaluate.getscore()

#else:
