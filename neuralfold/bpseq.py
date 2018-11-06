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
