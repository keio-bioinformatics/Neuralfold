class Decoder:
    def __init__(self):
        pass

    def dot_parenthesis(self, seq, pair):
        if len(pair) == 0:
            return '.'*len(seq)

        elif len(pair[0]) > 0 and type(pair[0][0]) is int: # nussinov
            return self.dot_parenthesis(seq, [pair])

        else: # ipknot
            parens = "()[]{}<>"
            assert len(pair) < len(parens)/2, "exceed max levels"
            y = ['.']*len(seq)
            for k, kpair in enumerate(pair):
                l, r = parens[k*2], parens[k*2+1]
                for i, j in kpair:
                    y[i], y[j] = l, r
            return "".join(y)
    