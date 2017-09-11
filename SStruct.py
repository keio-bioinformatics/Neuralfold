import numpy as np

class SStruct:
    def __init__(self,filename):
        self.filename = filename

    def load_FASTA(self):
        f = open(self.filename)
        line = f.readline()

        name = np.array([])
        seq = np.array([])
        structure = np.array([])

        while line:
            name=np.append(name,line.replace('\n' , ''))
            line = f.readline()

            seq=np.append(seq,line.replace('\n' , ''))
            line = f.readline()

            structure=np.append(structure,line.replace('\n' , ''))
            line = f.readline()

        f.close()
        return name,seq,structure
