import numpy as np
import sys
import SStruct
import Inference
args = sys.argv
filename = args[2]

sstruct = SStruct.SStruct(filename)
name,seq,structure = sstruct.load_FASTA()

if args[1]=='train':
    for s in seq:
        inference = Inference.Inference(s)
        BP = inference.ComputeInsideOutside()
        predicted_structure = inference.ComputePosterior(BP)
#elif args[1]=='test':

#else:
