import argparse

import chainer.functions as F
import numpy as np
import pulp
from chainer import Variable, link, optimizers, variable

from . import Decoder
from .nussinov import Nussinov

class IPknot(Decoder):
    def __init__(self, gamma=None, solver=None, solver_path=None, 
                    enable_stacking=False, no_approx_thresholdcut=False,
                    enable_inter_level_pk=False, dualdecomp_verbose=False,
                    dualdecomp_lr=0.01, dualdecomp_max_iter=100):
        self.gamma = gamma
        self.stacking = enable_stacking
        self.approx_cutoff = not no_approx_thresholdcut
        self.enable_inter_level_pk = enable_inter_level_pk
        self.verbose_dualdecomp = dualdecomp_verbose
        self.lr = dualdecomp_lr
        self.max_iter = dualdecomp_max_iter

        self.solver = None
        self.decode = self.decode_ip
        if solver == 'dualdecomp':
            self.decode = self.decode_dd
        elif solver == 'cplex':
            self.solver = pulp.CPLEX_CMD(path=solver_path, msg=False)
        elif solver == 'gurobi':
            self.solver = pulp.GUROBI_CMD(path=solver_path, msg=False)
        elif solver_path is not None:
            self.solver = pulp.COIN_CMD(path=solver_path, msg=False)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for IPknot')
        group.add_argument('--solver', 
                        help='IP solver to use for IPknot (default: dualdecomp)',
                        choices=('dualdecomp', 'coin', 'cplex', 'gurobi'),
                        type=str, default='dualdecomp')
        group.add_argument('--solver-path',
                        help='path of the solver executable',
                        type=str, default=None, metavar='PATH')
        group.add_argument('--enable-stacking', 
                        #help='enable stacking constrint', 
                        help=argparse.SUPPRESS,
                        action='store_true')
        group.add_argument('--no-approx-thresholdcut',
                        #help='approximate threshold cut',
                        help=argparse.SUPPRESS,
                        action='store_true')
        group.add_argument('--enable-inter-level-pk',
                        # help='enable constraints that ensure inter level pseudoknots'
                        help=argparse.SUPPRESS,
                        action='store_true')
        group.add_argument('--dualdecomp-lr', type=float,
                        help='learning rate for dual decompositon (default: 0.01)',
                        default=0.01, metavar='LR')
        group.add_argument('--dualdecomp-max-iter', type=int,
                        help='the maximum number of iteration of dual decomposition (default: 100)',
                        default=100, metavar='MAX_ITER')
        group.add_argument('--dualdecomp-verbose',
                        help=argparse.SUPPRESS,
                        action='store_true')


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('solver', 'solver-path', 'enable_stacking', 'no_approx_thresholdcut',
                        'enable_inter_level_pk', 'dualdecomp_verbose', 'dualdecomp_lr', 'dualdecomp_max_iter')
        return {p: getattr(args, p, None) for p in hyper_params if getattr(args, p, None) is not None}

    
    def decode_ip(self, seq, bpp, gamma=None, margin=None, allowed_bp='canonical', disable_th=False):
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
        if self.enable_inter_level_pk:
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


    def decode_dd(self, seq, bpp, gamma=None, margin=None, allowed_bp='canonical'):
        gamma = self.gamma if gamma is None else gamma
        bpp = bpp.array if isinstance(bpp, Variable) else bpp
        allowed_bp = self.allowed_basepairs(seq, allowed_bp)
        xp = np
        dtype = bpp.dtype
        K = len(gamma)
        N = len(seq)
        lagrangian = Lagrangian(K, N, self.lr, dtype=dtype, xp=xp, 
                        enable_inter_level_pk=self.enable_inter_level_pk, 
                        verbose=self.verbose_dualdecomp)

        for it in range(self.max_iter):
            if self.verbose_dualdecomp:
                print('iter:', it)
            y, _, c = lagrangian(seq, bpp, gamma, allowed_bp=allowed_bp, margin=margin)
            if c == 0:
                return y

        y_str = ['.']*len(seq)
        y_ret = []
        for y_p in y:
            y_p_ret = []
            for i, j in y_p:
                if y_str[i] != '.' and y_str[j] != '.':
                    y_str[i], y_str[j] = '(', ')'
                    y_p_ret.append((i, j))
            y_ret.append(y_p)

        return y_ret


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


class Lagrangian:
    def __init__(self, K, N, lr=0.01, momentum=0.9, dtype=np.float32, xp=np, 
                    enable_inter_level_pk=False, verbose=False):
        self.K = K
        self.N = N
        self.xp = xp
        self.lr = lr
        self.momentum = momentum
        self.verbose = verbose
        self.enable_inter_level_pk = enable_inter_level_pk
        self.c_sum, self.it = 0, 0
        self.mu = xp.zeros((N,), dtype=dtype)
        self.mu_v = xp.zeros((N,), dtype=dtype)
        if self.enable_inter_level_pk:
            self.xi = xp.zeros((K, K, N, N), dtype=dtype)
            self.xi_v = xp.zeros((K, K, N, N), dtype=dtype)
        self.nussinov = Nussinov()


    def calc_penalty(self):
        xp, dtype = self.xp, self.mu.dtype
        K, N = self.K, self.N

        penalty = xp.zeros((K, N, N), dtype=dtype)
        # constraint 1:
        mu_temp = xp.tile(self.mu, (K, N, 1))
        penalty += - (mu_temp + xp.transpose(mu_temp, axes=(0, 2, 1)))
        # constraint 3:
        if self.enable_inter_level_pk:
            for p in range(K):
                for q in range(p):
                    for l in range(N):
                        for k in range(l):
                            penalty[q, :k, k+1:l] += self.xi[p, q, k, l]
                            penalty[q, k+1:l, l+1:] += self.xi[p, q, k, l]
                    penalty[p, :, :] += -self.xi[p, q, :, :]

        return penalty, xp.sum(self.mu)

    
    def update(self, y):
        xp = self.xp
        K, N = self.K, self.N

        ymat = xp.zeros((K, N, N), dtype=np.bool)
        for p in range(K):
            for i, j in y[p]:
                ymat[p, i, j] = True
        y = ymat

        c = 0
        # constraint 1
        mu_grad = xp.ones_like(self.mu) - (y.sum(axis=(0, 2)) + y.sum(axis=(0, 1)))
        c += xp.sum(mu_grad < 0)
        # constraint 2
        if self.enable_inter_level_pk:
            xi_grad = xp.zeros_like(self.xi)
            for p in range(K):
                for q in range(p):
                    for l in range(N):
                        for k in range(l):
                            xi_grad[p, q, k, l] += y[q, :k, k+1:l].sum()
                            xi_grad[p, q, k, l] += y[q, k+1:l, l+1:].sum()
                    xi_grad[p, q, :, :] -= y[p, :, :]
            c += xp.sum(xi_grad < 0)
        self.c_sum += c
        self.it += 1
        if self.momentum is None:
            lr = self.lr / self.it
            #lr = self.lr / np.sqrt(1+self.c_sum)
            #lr = self.lr / (1+self.c_sum)

            # update by subgradients
            self.mu = self.mu - lr * mu_grad
            self.mu = xp.where(self.mu>0., self.mu, 0.)
            if self.enable_inter_level_pk:
                self.xi = self.xi - lr * xi_grad
                self.xi = xp.where(self.xi>0., self.xi, 0.)
        else:
            lr = self.lr
            # update by subgradients
            self.mu_v *= self.momentum
            self.mu_v -= self.lr * mu_grad
            self.mu += self.mu_v
            self.mu = xp.where(self.mu>0., self.mu, 0.)
            if self.enable_inter_level_pk:
                self.xi_v *= self.momentum
                self.xi_v -= self.lr * xi_grad
                self.xi += self.xi_v
                self.xi = xp.where(self.xi>0., self.xi, 0.)

        if self.verbose:
            print('violated constraints: {}, lr: {}'.format(c, lr))

        return c

    
    def __call__(self, seq, bpp, gamma, allowed_bp=None, margin=None):
        K = self.K
        nussinov = self.nussinov

        # solve decomposed problem
        penalty, constant = self.calc_penalty()
        s = constant
        y = []
        for p in range(K):
            margin_p = penalty[p] if margin is None else margin+penalty[p]
            y_p = nussinov.decode(seq, bpp, gamma=gamma[p], allowed_bp=allowed_bp, margin=margin_p)
            y.append(y_p)
            s += nussinov.calc_score(seq, bpp, y_p, gamma=gamma[p], margin=margin_p)
        del penalty
        s = s[0]

        c = self.update(y)

        if self.verbose:
            print('Score: {:.3f}'.format(s))
            print(seq)
            for p, y_p in enumerate(y):
                print(nussinov.dot_parenthesis(seq, y_p, k=p))
            print()

        return y, s, c


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
