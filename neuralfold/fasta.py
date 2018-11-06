import io

def load(filename):
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
                structure = parse_parenthesis(line.rstrip())
                structure_set.append(structure)
                line = f.readline()

        f.close()

    return name_set, seq_set, structure_set

def parse_parenthesis(line, paren=r'()'):
    i=0
    stack = []
    structure = []
    for a in line:
        if a == paren[0]:
            stack.append(i)
        elif a == paren[1]:
            structure.append((stack[-1], i))
            stack.pop()
        i+=1
    return structure
