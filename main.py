import numpy as np
import sys
import SStruct

args = sys.argv
filename = args[2]

sstruct = SStruct.SStruct(filename)
name,seq,structure = sstruct.load_FASTA()

if args[1]=='train':

#elif args[1]=='test':

else:
