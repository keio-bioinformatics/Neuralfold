def load_seq(filename):
    '''load sequences with their structures from file.
    FASTA format and BPSEQ format are supported.
    '''
    if not isinstance(filename, list):
        filename = [filename]

    name_set = []
    seq_set = []
    str_set = []
    for f in filename:
        if isinstance(f, str):
            f = open(f)
        
        line = f.readline()
        f.seek(0)
        ret = load_fasta(f) if line[0] == '>' else load_bpseq(f)
        name_set += ret[0]
        seq_set += ret[1]
        str_set += ret[2]

    return name_set, seq_set, str_set


def load_fasta(filename):
    name_set = []
    seq_set = []
    structure_set = []

    if not isinstance(filename, list):
        filename = [filename]

    for f in filename:
        if isinstance(f, str):
            f = open(f)

        line = f.readline()
        while line:
            name_set.append(line.rstrip())
            line = f.readline()
            seq_set.append(line.rstrip())
            line = f.readline()
            if not line:
                break
            if line[0] != ">":
                structure = []
                for paren in (r'()', r'[]', r'{}', r'<>'):
                    structure += parse_parenthesis(line.rstrip(), paren)
                line = f.readline()

        f.close()

    return name_set, seq_set, structure_set

def parse_parenthesis(line, paren=r'()'):
    stack = []
    structure = []
    for i, a in enumerate(line):
        if a == paren[0]:
            stack.append(i)
        elif a == paren[1]:
            structure.append((stack[-1], i))
            stack.pop()
    return structure

def load_bpseq(filename):
    name_set = []
    seq_set = []
    structure_set = []

    if not isinstance(filename, list):
        filename = [filename]

    for f in filename:
        if isinstance(f, str):
            f = open(f)
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
