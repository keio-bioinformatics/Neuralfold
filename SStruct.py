import numpy as np

class SStruct:
    def __init__(self,filename,train=False):
        self.filename = filename
        #reverse
        self.train = train

    def load_FASTA(self):
        # print(self.filename)
        # f = open(self.filename.name)
        # f = self.filename[0]

        name_set = []
        seq_set = []
        structure_set = []

        # Check if there are multiple files
        if not isinstance(self.filename, list):
            self.filename = [self.filename]

        for f in self.filename:
            line = f.readline()

            while line:
                name_set.append(line.replace('\n' , ''))
                if self.train:
                    name_set.append(line.replace('\n' , ''))

                line = f.readline()

                seq_set.append(line.replace('\n' , ''))
                if self.train:
                    seq_set.append(line.replace('\n' , '')[::-1])

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
                    if self.train:
                        length = len(seq_set[-1])
                        structure_set.append([(length-i[1]-1, length-i[0]-1) for i in structure])
                    # structure.append(line.replace('\n' , ''))

                    line = f.readline()
                else:
                    continue

            f.close()
            # print(structure_set)
            # for seq in seq_set:
            # print(len(max(seq_set,key=len)))
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

        return name_set, seq_set, structure_set
