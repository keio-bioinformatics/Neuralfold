import numpy as np
import pulp
from chainer import Variable

from . import Decoder


class IPknot(Decoder):
    def __init__(self, gamma=None):
        self.gamma = gamma
    
    def decode(self, bpp, gamma=None, margin=None, allowed_bp=None, disable_th=False):
        '''IPknot-style decoding algorithm    
        '''
        gamma = self.gamma if gamma is None else gamma
        prob = pulp.LpProblem("IPknot", pulp.LpMaximize)
        seqlen = len(bpp)
        nlevels = len(gamma)
        bpp = bpp.data if type(bpp) is Variable else bpp

        if allowed_bp is None:
            allowed_bp = np.ones(bpp.shape, dtype=np.bool)
        elif type(allowed_bp) is list:
            a = np.zeros(bpp.shape, dtype=np.bool)
            for l, r in allowed_bp:
                a[l, r] = a[r, l] = True
            allowed_bp = a
        assert type(allowed_bp) is np.ndarray
        assert allowed_bp.shape == bpp.shape

        # accelerate by 'at most gamma+1' candidates for each base
        if int(max(gamma))<seqlen:
            top_n = int(max(gamma))
            bpp_tri = np.triu(bpp)
            bpp = bpp_tri + np.transpose(bpp_tri)
            sorted_bp_idx = np.argsort(bpp, axis=1)[:, ::-1]
            enabled_bp = np.zeros(bpp.shape, dtype=np.bool)
            for i in range(seqlen):
                th_i = bpp[i, sorted_bp_idx[i, top_n]]
                enabled_bp[i, bpp[i, :]>=th_i] = True
            allowed_bp &= enabled_bp & np.transpose(enabled_bp)

        # variables and objective function
        x = [[[None for i in range(seqlen)] for j in range(seqlen)] for k in range(nlevels)]
        obj = []
        th = 0 if disable_th else 1
        for k in range(nlevels):
            for j in range(seqlen):
                for i in range(j):
                    if allowed_bp[i, j]:
                        s = (gamma[k]+1) * bpp[i, j] - th
                        if s > 0:
                            if margin is not None:
                                s += margin[i, j]
                            x[k][i][j] = x[k][j][i] = pulp.LpVariable('x[%d][%d][%d]' % (k, i, j), 0, 1, 'Binary')
                            obj.append(s * x[k][i][j])
        prob += pulp.lpSum(obj)

        # constraints 1
        for i in range(seqlen):
            prob += pulp.lpSum([x[k][i][j] for j in range(seqlen) for k in range(nlevels) if x[k][i][j] is not None]) <= 1

        # constraints 2
        for k in range(nlevels):
            for j2 in range(seqlen):
                for j1 in range(j2):
                    for i2 in range(j1):
                        if x[k][i2][j2] is not None:
                            for i1 in range(i2):
                                if x[k][i1][j1] is not None:
                                    prob += pulp.lpSum(x[k][i1][j1] + x[k][i2][j2]) <= 1

        # constraints 3
        for k2 in range(nlevels):
            for j2 in range(seqlen):
                for i2 in range(j2):
                    if x[k2][i2][j2] is not None:
                        for k1 in range(k2):
                            # o = 0
                            c1 = [x[k1][i1][j1] for i1 in range(i2+1, j2) for j1 in range(i2-1) if x[k1][j1][j1] is not None]
                            c2 = [x[k1][i1][j1] for i1 in range(i2+1, j2) for j1 in range(j2+1, seqlen) if x[k1][i1][j1] is not None]
                            prob += pulp.lpSum(c1+c2) >= x[k2][i2][j2]

        # solve the IP problem
        # start_ip = time()
        prob.solve()

        pair = [ [] for _ in range(nlevels) ]
        for k in range(nlevels):
            for j in range(seqlen):
                for i in range(j):
                    if x[k][i][j] is not None and x[k][i][j].value() > 0.5:
                        pair[k].append((i,j))
        return pair
    
    def calc_score(self, bpp, pair, gamma=None, margin=None):
        gamma = self.gamma if gamma is None else gamma
        s = np.zeros((1,1), dtype=np.float32)
        if type(bpp) is Variable:
            s = Variable(s)

        if len(pair) == 0:
            return s.reshape(1,)
        elif len(gamma) == len(pair) and type(pair[0]) is list:
            pass
        else:
            pair = self.decode(bpp, gamma, margin=margin, allowed_bp=pair, disable_th=True)    

        for k, kpair in enumerate(pair):
            for i, j in kpair:
                s += (gamma[k] + 1) * bpp[i, j] - 1
                if margin is not None:
                    s += margin[i, j]

        return s.reshape(1,)

if __name__ == '__main__':
    import sys
    import RNA

    def read_fasta(fasta):
        def split_seq(l):
            l = l.split("\n")
            name = l.pop(0)
            seq = "".join(l)
            return {"name": name, "seq": seq}

        with open(fasta, "r") as f:
            fastas = f.read().split(">")
            fastas.remove("")
            return [split_seq(l) for l in fastas]
    
    ipknot = IPknot(gamma=[9.0, 9.0])
    for fa in read_fasta(sys.argv[1]):
        rna = RNA.fold_compound(fa['seq'])
        rna.pf()
        print(">"+fa['name'])
        print(fa['seq'])
        bpp = np.array(rna.bpp())
        pred = ipknot.decode(bpp[1:, 1:])
        print(ipknot.dot_parenthesis(fa['seq'], pred))
        # pred = ipknot.decode(bpp[1:, 1:], [9.0])
        # print(ipknot.dot_parenthesis(fa['seq'], pred))
