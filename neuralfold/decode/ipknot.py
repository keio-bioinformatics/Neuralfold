import argparse

import chainer.functions as F
import numpy as np
import pulp
from chainer import Variable, link, optimizers, variable

from . import Decoder
from .nussinov import Nussinov


class IPknot(Decoder):
    def __init__(self, gamma=None, no_stacking=False, no_approx_thresholdcut=False,
                    cplex=False, cplex_path=None, gurobi=False, gurobi_path=None):
        self.gamma = gamma
        self.stacking = not no_stacking
        self.approx_cutoff = not no_approx_thresholdcut
        self.solver = None
        if cplex:
            self.solver = pulp.CPLEX_CMD(path=cplex_path, msg=False)
        if gurobi:
            self.solver = pulp.GUROBI_CMD(path=gurobi_path, msg=False)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for IPknot')
        group.add_argument('--no-stacking', 
                        help='no stacking constrint', 
                        action='store_true')
        group.add_argument('--cplex',
                        help='use CPLEX',
                        action='store_true')
        group.add_argument('--cplex-path',
                        help='path of CPLEX executable',
                        type=str, default=None)
        group.add_argument('--gurobi',
                        help='use Gurobi',
                        action='store_true')
        group.add_argument('--gurobi-path',
                        help='path of Gurobi executable',
                        type=str, default=None)
        group.add_argument('--no-approx-thresholdcut',
                        #help='approximate threshold cut',
                        help=argparse.SUPPRESS,
                        action='store_true')


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('cplex', 'cplex_path', 'gurobi', 'gurobi_path', 'no_stacking', 'no_approx_thresholdcut')
        return {p: getattr(args, p, None) for p in hyper_params if getattr(args, p, None) is not None}

    
    def decode(self, seq, bpp, gamma=None, margin=None, allowed_bp='canonical', disable_th=False):
        '''IPknot-style decoding algorithm    
        '''
        gamma = self.gamma if gamma is None else gamma
        prob = pulp.LpProblem("IPknot", pulp.LpMaximize)
        N = len(seq)
        K = len(gamma)
        bpp = bpp.data if isinstance(bpp, Variable) else bpp

        allowed_bp = self.allowed_basepairs(seq, allowed_bp)
        assert isinstance(allowed_bp, np.ndarray)
        assert allowed_bp.shape == bpp.shape

        # accelerate by 'at most gamma+1' candidates for each base
        if self.approx_cutoff and int(max(gamma))<N:
            top_n = int(max(gamma))
            bpp_tri = np.triu(bpp)
            bpp = bpp_tri + np.transpose(bpp_tri)
            bpp[allowed_bp==False] = -1e10 # if bpp is correctly trained, this line may not be needed.
            sorted_bp_idx = np.argsort(bpp, axis=1)[:, ::-1]
            enabled_bp = np.zeros_like(bpp, dtype=np.bool)
            for i in range(N):
                th_i = bpp[i, sorted_bp_idx[i, top_n]]
                enabled_bp[i, bpp[i, :]>=th_i] = True
                # for j in sorted_bp_idx[i, :top_n]:
                #     enabled_bp[i, j] = True
            allowed_bp &= enabled_bp & np.transpose(enabled_bp)

        # if self.approx_cutoff and int(max(gamma))<N:
        #     bpp_tri = np.triu(bpp)
        #     bpp = bpp_tri + np.transpose(bpp_tri)
        #     bpp[bpp<0.01] = 0.01
        #     bpp[bpp>0.99] = 0.99
        #     bpp = np.log(bpp) - np.log(1-bpp)
        #     bpp = F.softmax(bpp, axis=1).data
        #     th = 1./(max(gamma)+1)
        #     #print(np.sum(bpp>th, axis=1))
        #     allowed_bp &= bpp>th


        # variables and objective function
        x = [[[None for _ in range(N)] for _ in range(N)] for _ in range(K)]
        x_up_idx = [[[] for _ in range(N)] for _ in range(K)]
        x_do_idx = [[[] for _ in range(N)] for _ in range(K)]        
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
                            x_up_idx[k][j].append(i)
                            x_do_idx[k][i].append(j)
        prob += pulp.lpSum(obj)

        # constraints 1
        for i in range(N):
            prob += pulp.lpSum([x[k][i][j] for k in range(K) for j in x_up_idx[k][i]+x_do_idx[k][i]]) <= 1

        # constraints 2
        for k in range(K):
            for j2 in range(N):
                for j1 in range(j2):
                    for i2 in x_up_idx[k][j2]:
                        assert x[k][i2][j2] is not None
                        if i2 < j1:
                            for i1 in x_up_idx[k][j1]:
                                if i1 < i2:
                                    assert x[k][i1][j1] is not None
                                    prob += pulp.lpSum(x[k][i1][j1] + x[k][i2][j2]) <= 1

        # constraints 3
        for k2 in range(K):
            for j2 in range(N):
                for i2 in x_up_idx[k2][j2]:
                    assert x[k2][i2][j2] is not None
                    for k1 in range(k2):
                        c1 = [x[k1][i1][j1] for i1 in range(0, i2-1) for j1 in x_do_idx[k1][i1] if i2 < j1 and j1 < j2]
                        c2 = [x[k1][i1][j1] for j1 in range(j2+1, N) for i1 in x_up_idx[k1][j1] if i2 < i1 and i1 < j2]
                        prob += pulp.lpSum(c1+c2) >= x[k2][i2][j2]

        # constraints 4: stacking
        if self.stacking:
            for k in range(K):
                x_up = [ None ] * N
                x_do = [ None ] * N
                for i in range(N):
                    x_up[i] = pulp.lpSum([ x[k][j][i] for j in x_up_idx[k][i] ])
                    x_do[i] = pulp.lpSum([ x[k][i][j] for j in x_do_idx[k][i] ])
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
        prob.solve(self.solver)

        pair = [ [] for _ in range(K) ]
        for k in range(K):
            for j in range(N):
                for i in x_up_idx[k][j]:
                    assert x[k][i][j] is not None
                    if x[k][i][j].value() > 0.5:
                        pair[k].append((i,j))
        return pair


    def decode_dd(self, seq, bpp, gamma=None, margin=None, allowed_bp='canonical', max_iter=100):
        verbose = True
        gamma = self.gamma if gamma is None else gamma
        bpp = bpp.array if isinstance(bpp, Variable) else bpp
        allowed_bp = self.allowed_basepairs(seq, allowed_bp)
        K = len(gamma)
        N = len(seq)
        nussinov = Nussinov()

        lag = Lagrangian(K, N, dtype=bpp.dtype)
        optimizer = optimizers.SGD(lr=0.01).setup(lag)

        for it in range(max_iter):
            if verbose:
                print('iter:', it)
            penalty = lag()
            s = 0
            y = []
            for p in range(K):
                pmargin = penalty[p] if margin is None else margin+penalty[p]
                y_p = nussinov.decode(seq, bpp, gamma=gamma[p], allowed_bp=allowed_bp, margin=pmargin.array)
                y.append(y_p)
                s += nussinov.calc_score(seq, bpp, y_p, gamma=gamma[p], margin=pmargin)
            s += lag.constants()
            if verbose:
                print(seq)
                for y_p in y:
                    print(nussinov.dot_parenthesis(seq, y_p))

            lag.cleargrads()
            s.backward()
            c = lag.count_violates()
            if c == 0:
                break
            if verbose:
                print('violated constraints: {}'.format(c))
                print()
            optimizer.update()
            print(np.sum(lag.mu.data>0), np.sum(lag.mu.data<0), np.sum(lag.xi.data>0), np.sum(lag.xi.data<0))
            lag.clip()
            print(np.sum(lag.mu.data>0), np.sum(lag.mu.data<0), np.sum(lag.xi.data>0), np.sum(lag.xi.data<0))            
        
        return y


    def calc_score_0(self, seq, bpp, pair, gamma=None, margin=None):
        gamma = self.gamma if gamma is None else gamma
        s = np.zeros((1,1), dtype=np.float32)
        if isinstance(bpp, Variable):
            s = Variable(s)

        if len(pair) == 0:
            return s.reshape(1,)
        elif len(gamma) == len(pair) and isinstance(pair[0], list):
            pass
        else:
            pair = self.decode(seq, bpp, gamma, margin=margin, allowed_bp=pair, disable_th=True)    

        for k, kpair in enumerate(pair):
            for i, j in kpair:
                s += (gamma[k] + 1) * bpp[i, j] - 1
                if margin is not None:
                    s += margin[i, j]

        return s.reshape(1,)


    def calc_score(self, seq, bpp, pair, gamma=None, margin=None):
        gamma = self.gamma if gamma is None else gamma
        s = np.zeros((1,1), dtype=np.float32)
        if isinstance(bpp, Variable):
            s = Variable(s)

        if len(pair) == 0:
            return s.reshape(1,)
        elif len(gamma) == len(pair) and isinstance(pair[0], list):
            temp = []
            for kpair in pair:
                temp.extend(kpair)
            pair = temp

        for i, j in pair:
            s += (gamma[0] + 1) * bpp[i, j] - 1
            if margin is not None:
                s += margin[i, j]

        return s.reshape(1,)


class Lagrangian(link.Link):
    def __init__(self, K, N, dtype=np.float32):
        super(Lagrangian, self).__init__()
        xp = self.xp
        with self.init_scope():
            self.mu = variable.Parameter(xp.zeros((N,), dtype=dtype))
            self.xi = variable.Parameter(xp.zeros((K, K, N, N), dtype=dtype))


    def __call__(self):
        K = self.xi.shape[0]
        N = self.xi.shape[2]
        xp = self.xp
        dtype = self.xi.dtype
        penalty = Variable(xp.zeros((K, N, N), dtype=dtype))

        # constraint 1:
        mu_temp = F.tile(self.mu, (K, N, 1))
        penalty += - mu_temp - F.transpose(mu_temp, axes=(0, 2, 1))

        # constraint 3:
        for p in range(K):
            for q in range(p):
                for l in range(N):
                    for k in range(l):
                        x = xp.zeros((K, N, N), dtype=dtype)
                        x[q, :k, k+1:l] = 1
                        x[q, k+1:l, l+1:] = 1
                        penalty += x * self.xi[p, q, k, l]
                x = xp.zeros((K, N, N), dtype=dtype)
                x[p, :, :] = -1
                penalty += x * self.xi[:, q, :, :]

        return penalty


    def constants(self):
        return F.sum(self.mu)


    def mask(self, p, th=0.):
        xp = self.xp
        return F.where(p.array>th, p, xp.zeros_like(p.array))


    def clip(self):
        self.mu = self.mask(self.mu)
        self.xi = self.mask(self.xi)

    
    def count_violates(self):
        xp = self.xp
        c = xp.sum(self.mu.grad < 0) + xp.sum(self.xi.grad < 0)
        return c


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
        pred = ipknot.decode(fa['seq'], bpp[1:, 1:])
        print(ipknot.dot_parenthesis(fa['seq'], pred))
        # pred = ipknot.decode(bpp[1:, 1:], [9.0])
        # print(ipknot.dot_parenthesis(fa['seq'], pred))
