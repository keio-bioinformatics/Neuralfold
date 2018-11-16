import numpy as np

class Decoder:
    def __init__(self):
        pass

    def dot_parenthesis(self, seq, pair):
        if len(pair) == 0:
            return '.'*len(seq)

        elif len(pair[0]) > 0 and type(pair[0][0]) is int: # nussinov
            return self.dot_parenthesis(seq, [pair])

        else: # ipknot
            parens = r'()[]{}<>'
            assert len(pair) < len(parens)/2, "exceed max levels"
            y = ['.']*len(seq)
            for k, kpair in enumerate(pair):
                l, r = parens[k*2], parens[k*2+1]
                for i, j in kpair:
                    y[i], y[j] = l, r
            return "".join(y)


    def allowed_basepairs(self, seq, allowed_bp):
        N = len(seq)
        if allowed_bp is None:
            allowed_bp = np.ones((N, N), dtype=np.bool)
        elif allowed_bp == 'canonical':
            allowed_bp = np.zeros((N, N), dtype=np.bool)
            canonicals = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G' )}
            seq = seq.upper()
            for j in range(N):
                for i in range(j):
                    if (seq[i], seq[j]) in canonicals:
                        allowed_bp[i, j] = allowed_bp[j, i] = True
        elif isinstance(allowed_bp, list):
            a = np.zeros((N, N), dtype=np.bool)
            for i, j in allowed_bp:
                a[i, j] = a[j, i] = True
            allowed_bp = a

        return allowed_bp
    