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

        name_set = []
        seq_set = []
        structure_set = []

        while line:
            name_set.append(line.replace('\n' , ''))
            line = f.readline()

            seq_set.append(line.replace('\n' , ''))
            line = f.readline()
            if not line:
                break
            if line[0] != ">":
                i=0
                stack = []
                structure = []
                for a in line:
                    if a == "(":
                        stack.append(i)
                    elif a == ")":
                        structure.append((stack[-1], i))
                        stack.pop()
                    i+=1
                structure_set.append(structure)
                structure.append(line.replace('\n' , ''))
                if self.small_structure:
                    if len(line)>50:
                        name_set.pop()
                        seq_set.pop()
                        structure_set.pop()

                line = f.readline()
            else:
                continue

        f.close()
        return name_set, seq_set, structure_set


    def load_BPseq(self):
        # f = open(self.filename)
        name_set = []
        seq_set = []
        structure_set = []
        for f in self.filename:
            line = f.readline()
            seq = []
            structure = []
            while line:
                line = line.split()
                seq.append(line[1])
                if line[2] != "0" and int(line[0]) < int(line[2]):
                    structure.append((int(line[0])-1, int(line[2])-1))
                line = f.readline()

            name_set.append(f.name)
            seq_set.append(''.join(seq))
            structure_set.append(structure)
            if self.small_structure:
                if len(seq_set[-1])>40:
                    name_set.pop()
                    seq_set.pop()
                    structure_set.pop()

        return name_set, seq_set, structure_set
