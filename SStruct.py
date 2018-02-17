import numpy as np

class SStruct:
    def __init__(self,filename, small_structure):
        self.filename = filename
        self.small_structure = small_structure

    def load_FASTA(self):
        # print(self.filename)
        # f = open(self.filename.name)
        # f = self.filename[0]
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
            structure=np.append(structure,line.replace('\n' , ''))
            if self.small_structure:
                if len(line)>50:
                    name_set = np.delete(name_set, -1)
                    seq_set = np.delete(seq_set,-1)
                    structure_set.pop(-1)

            line = f.readline()
        f.close()
        return name_set, seq_set, structure_set


    def load_BPseq(self):
        # f = open(self.filename)
        name_set = np.array([])
        seq_set = np.array([])
        structure_set = []
        for f in self.filename:
            # print(f)
            line = f.readline()
            seq=[]
            structure = np.empty((0, 2),dtype=np.int16)
            while line:
                line = line.split()
                seq.append(line[1])
                if line[2] != "0" and int(line[0]) < int(line[2]):
                    structure = np.append(structure, [[int(line[0])-1,int(line[2])-1]], axis=0)
                line = f.readline()

            seq_set = np.append(seq_set, ''.join(seq))
            structure_set.append(structure)
            if self.small_structure:
                if len(seq_set[-1])>40:
                    seq_set = np.delete(seq_set,-1)
                    structure_set.pop(-1)

        # print(len(seq_set))
        # print(seq_set)
        # print(structure_set)
        return name_set, seq_set, structure_set
