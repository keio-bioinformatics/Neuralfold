import numpy as np

class SStruct:
    def __init__(self,filename):
        self.filename = filename

    def load_FASTA(self):
        # f = open(self.filename)
        f = self.filename
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
            stack = np.array([],dtype=np.int16)
            structure = np.empty((0, 2),dtype=np.int16)
            for a in line:
                if a == "(":
                    stack = np.append(stack , i)
                elif a == ")":
                    structure =  np.append(structure, [[stack[-1], i]], axis=0)
                    stack = np.delete(stack, -1)
                i+=1
            structure_set.append(structure)
            #structure=np.append(structure,line.replace('\n' , ''))
            if len(line)>50:
                name_set = np.delete(name_set, -1)
                seq_set = np.delete(seq_set,-1)
                structure_set.pop(-1)


            line = f.readline()

        f.close()

        return name_set, seq_set, structure_set
