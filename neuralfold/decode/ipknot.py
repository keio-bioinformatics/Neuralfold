import numpy as np
import pulp
from chainer import Variable

from . import Decoder


class IPknot(Decoder):
    def __init__(self, gamma=None, stacking=True):
        self.gamma = gamma
        self.stacking = stacking


    def decode(self, bpp, gamma=None, margin=None, allowed_bp=None, disable_th=False):
        '''IPknot-style decoding algorithm    
        '''
        gamma = self.gamma if gamma is None else gamma
        prob = pulp.LpProblem("IPknot", pulp.LpMaximize)
        N = len(bpp)
        K = len(gamma)
        bpp = bpp.data if isinstance(bpp, Variable) else bpp

        if allowed_bp is None:
            allowed_bp = np.ones_like(bpp, dtype=np.bool)
        elif isinstance(allowed_bp, list):
            a = np.zeros_like(bpp, dtype=np.bool)
            for i, j in allowed_bp:
                a[i, j] = a[j, i] = True
            allowed_bp = a
        assert isinstance(allowed_bp, np.ndarray)
        assert allowed_bp.shape == bpp.shape

        # accelerate by 'at most gamma+1' candidates for each base
        if int(max(gamma))<N:
            top_n = int(max(gamma))
            bpp_tri = np.triu(bpp)
            bpp = bpp_tri + np.transpose(bpp_tri)
            sorted_bp_idx = np.argsort(bpp, axis=1)[:, ::-1]
            enabled_bp = np.zeros_like(bpp, dtype=np.bool)
            for i in range(N):
                th_i = bpp[i, sorted_bp_idx[i, top_n]]
                enabled_bp[i, bpp[i, :]>=th_i] = True
            allowed_bp &= enabled_bp & np.transpose(enabled_bp)

        # variables and objective function
        x = [[[None for i in range(N)] for j in range(N)] for k in range(K)]
        x_i = [[[] for j in range(N)] for k in range(K)]
        x_j = [[[] for i in range(N)] for k in range(K)]        
        obj = []
        th = 0 if disable_th else 1
        for k in range(K):
            for j in range(N):
                for i in range(j):
                    if allowed_bp[i, j]:
                        s = (gamma[k]+1) * bpp[i, j] - th
                        if s > 0:
                            if margin is not None:
                                s += margin[i, j]
                            x[k][i][j] = x[k][j][i] = pulp.LpVariable('x[%d][%d][%d]' % (k, i, j), 0, 1, 'Binary')
                            obj.append(s * x[k][i][j])
                            x_i[k][j].append(i)
                            x_j[k][i].append(j)
        prob += pulp.lpSum(obj)

        # constraints 1
        for i in range(N):
            prob += pulp.lpSum([x[k][i][j] for j in x_i[k][i]+x_j[k][i] for k in range(K)]) <= 1

        # constraints 2
        for k in range(K):
            for j2 in range(N):
                for j1 in range(j2):
                    for i2 in x_i[k][j2]:
                        assert x[k][i2][j2] is not None
                        if i2 < j1:
                            for i1 in x_i[k][j1]:
                                if i1 < i2:
                                    assert x[k][i1][j1] is not None
                                    prob += pulp.lpSum(x[k][i1][j1] + x[k][i2][j2]) <= 1

        # constraints 3
        for k2 in range(K):
            for j2 in range(N):
                for i2 in x_i[k2][j2]:
                    assert x[k2][i2][j2] is not None
                    for k1 in range(k2):
                        c1 = [x[k1][i1][j1] for j1 in x_j[k1][i1] for i1 in range(0, i2-1) if i2 < j1 and j1 < j2]
                        c2 = [x[k1][i1][j1] for i1 in x_i[k1][j1] for j1 in range(j2+1, N) if i2 < i1 and i1 < j2]
                        prob += pulp.lpSum(c1+c2) >= x[k2][i2][j2]

        # constraints 4: stacking
        if self.stacking:
            for k in range(K):
                x_up = [ None ] * N
                x_do = [ None ] * N
                for i in range(N):
                    x_up[i] = pulp.lpSum([ x[k][j][i] for j in x_i[k][i] ])
                    x_do[i] = pulp.lpSum([ x[k][i][j] for j in x_j[k][i] ])
                for i in range(N):
                    c_up = [-x_up[i]]
                    c_do = [-x_do[i]]
                    if i>0:
                        c_up.append(x_up[i-1])
                        c_do.append(x_do[i-1])
                    if i+1<N:
                        c_up.append(x_up[i+1])
                        c_do.append(x_do[i+1])
                    prob += pulp.lpSum(c_up) >= 0
                    prob += pulp.lpSum(c_do) >= 0

        # solve the IP problem
        prob.solve()

        pair = [ [] for _ in range(K) ]
        for k in range(K):
            for j in range(N):
                for i in x_i[k][j]:
                    assert x[k][i][j] is not None
                    if x[k][i][j].value() > 0.5:
                        pair[k].append((i,j))
        return pair


    def calc_score(self, bpp, pair, gamma=None, margin=None):
        gamma = self.gamma if gamma is None else gamma
        s = np.zeros((1,1), dtype=np.float32)
        if isinstance(bpp, Variable):
            s = Variable(s)

        if len(pair) == 0:
            return s.reshape(1,)
        elif len(gamma) == len(pair) and isinstance(pair[0], list):
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
