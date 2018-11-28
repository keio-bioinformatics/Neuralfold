import numpy as np

class Decoder:
    def __init__(self):
        pass

    def dot_parenthesis(self, seq, y, k=0):
        if len(y) == 0:
            return '.'*len(seq)

        elif len(y[0]) > 0 and type(y[0][0]) is int: # nussinov
            return self.dot_parenthesis(seq, [y], k)

        else: # ipknot
            parens = r'()[]{}<>'
            p_max = len(parens)
            assert len(y) < p_max, "exceed max levels"
            y_str = ['.']*len(seq)
            for p, y_p in enumerate(y):
                p = (p + k) % p_max
                l, r = parens[p*2], parens[p*2+1]
                for i, j in y_p:
                    y_str[i], y_str[j] = l, r
            return "".join(y_str)


    def allowed_basepairs(self, seq, allowed_bp):
        N = len(seq)
        if isinstance(allowed_bp, np.ndarray):
            pass    
        elif allowed_bp is None:
            allowed_bp = np.ones((N, N), dtype=np.bool)
        elif isinstance(allowed_bp, str) and allowed_bp == 'canonical':
            allowed_bp = np.zeros((N, N), dtype=np.bool)
            canonicals = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G' )}
            seq = seq.upper()
            for j in range(N):
                for i in range(j):
                    if (seq[i], seq[j]) in canonicals and j-i>2:
                        allowed_bp[i, j] = allowed_bp[j, i] = True
        elif isinstance(allowed_bp, list):
            a = np.zeros((N, N), dtype=np.bool)
            for i, j in allowed_bp:
                a[i, j] = a[j, i] = True
            allowed_bp = a

        return allowed_bp
    