import numpy as np
import pulp
from chainer import Variable

class IPknot:
    def __init__(self):
        pass
    
    def decode(self, bp, gamma, margin=None, allowed_bp=None, disable_th=False):
        '''IPknot-style decoding algorithm    
        '''
        prob = pulp.LpProblem("IPknot", pulp.LpMaximize)
        seqlen = len(bp)
        nlevels = len(gamma)

        if allowed_bp is None:
            allowed_bp = np.ones(bp.shape, dtype=np.bool)
        elif type(allowed_bp) is list:
            a = np.zeros(bp.shape, dtype=np.bool)
            for l, r in allowed_bp:
                a[l, r] = a[r, l] = True
            allowed_bp = a
        assert type(allowed_bp) is np.array
        assert allowed_bp.shape == bp.shape

        # accelerate by 'at most gamma+1' candidates for each base
        sorted_bp_idx = np.argsort(bp, axis=1)[:, ::-1]
        enabled_bp = np.zeros(bp.shape, dtype=np.bool)
        for i in range(seqlen):
            top_n = min(seqlen-1, int(max(gamma)+1))
            th_i = bp[i, sorted_bp_idx[i, top_n]]
            enabled_bp[i, bp[i, :]>=th_i] = True
        allowed_bp &= enabled_bp & np.transpose(enabled_bp)

        # variables and objective function
        x = [[[None for i in range(seqlen)] for j in range(seqlen)] for k in range(nlevels)]
        obj = []
        th = 0 if disable_th else 1
        for k in range(nlevels):
            for j in range(seqlen):
                for i in range(j):
                    if allowed_bp[i, j]:
                        s = (gamma[k]+1) * bp[i, j] - th
                        if margin is not None:
                            s += margin[i, j]
                        if type(s) is Variable:
                            s = s.data
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
    
    def calc_score(self, bp, gamma, pair, margin=None):
        s = np.zeros((1,1), dtype=np.float32)
        if type(bp) is Variable:
            s = Variable(s)

        if len(pair) == 0:
            return s.reshape(1,)
        elif len(gamma) == len(pair) and type(pair[0]) is list:
            pass
        else:
            pair = self.decode(bp, gamma, margin=margin, allowed_bp=pair, disable_th=True)    

        for k, kpair in enumerate(pair):
            for i, j in kpair:
                s += (gamma[k] + 1) * bp[i, j] - 1
                if margin is not None:
                    s += margin[i, j]

        return s.reshape(1,)

