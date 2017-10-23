import numpy as np

class SStruct:
    def __init__(self,filename):
        self.filename = filename

    def load_FASTA(self):
        f = open(self.filename)
        line = f.readline()

        name_set = np.array([])
        seq_set = np.array([])
        structure_set = []

        while line:
            name_set = np.append(name_set, line.replace('\n' , ''))
            line = f.readline()

            seq_set = np.append(seq_set, line.replace('\n' , ''))
            line = f.readline()

            i=0
            stack = np.array([])
            structure = np.empty((0, 2))
            for a in line:
                if a == "(":
                    stack = np.append(stack , i)
                elif a == ")":
                    structure =  np.append(structure, [[stack[-1], i]], axis=0)
                    stack = np.delete(stack, -1)
                i+=1
            structure_set.append(structure)
            #structure=np.append(structure,line.replace('\n' , ''))

            line = f.readline()

        f.close()

        return name_set, seq_set, structure_set
