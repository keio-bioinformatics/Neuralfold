import argparse

import cupy as cp
import numpy as np
from chainer import Variable

from . import Decoder


class Nussinov(Decoder):
    def __init__(self, gamma=None, simple_dp=False):
        self.gamma = gamma
        self.simple_dp = simple_dp


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for Nussinov')
        group.add_argument('--simple-dp', 
                        #help='Use simple dynamic programming', 
                        help=argparse.SUPPRESS,
                        action='store_true')

    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('simple_dp',)
        return {p: getattr(args, p, None) for p in hyper_params if getattr(args, p, None) is not None}


    def decode(self, seq, bpp, gamma=None, margin=None, allowed_bp='canonical'):
        '''Nussinov-style decoding algorithm    
        '''
        gamma = self.gamma if gamma is None else gamma
        N = len(bpp)
        bpp = bpp.data if type(bpp) is Variable else bpp
        if self.simple_dp:
            _, tr = self.build_dp(seq, bpp, gamma=gamma, allowed_bp=allowed_bp, margin=margin)
        else:
            _, tr = self.build_dp_v(seq, bpp, gamma=gamma, allowed_bp=allowed_bp, margin=margin)
        pair = self.traceback(tr, 0, N-1, [])
        return pair

    def build_dp(self, seq, bpp, gamma, margin=None, allowed_bp=None):
        N = len(seq)
        dp = np.zeros(bpp.shape, dtype=np.float32)
        tr = np.zeros(bpp.shape, dtype=np.int)

        allowed_bp = self.allowed_basepairs(seq, allowed_bp)

        # for j in range(N):
        #     for i in reversed(range(j)):
        for l in range(1, N):
            for i in range(N-l):
                j = i + l
                s = (gamma + 1) *  bpp[i, j] - 1
                if margin is not None:
                    s += margin[i, j]
                if i+1 < j:
                    dp[i, j] = dp[i+1, j]
                    tr[i, j] = -1
                if i < j-1 and dp[i, j] < dp[i, j-1]:
                    dp[i, j] = dp[i, j-1]
                    tr[i, j] = -2
                if i+1 < j-1 and allowed_bp[i, j] and dp[i, j] < dp[i+1, j-1] + s:
                    dp[i, j] = dp[i+1, j-1] + s
                    tr[i, j] = -3
                for k in range(i+1, j):
                    if dp[i, j] < dp[i, k] + dp[k+1, j]:
                        dp[i, j] = dp[i, k] + dp[k+1, j]
                        tr[i, j] = k-i

        return dp, tr


    def build_dp_v(self, seq, bpp, gamma, margin=None, allowed_bp=None):
        N = len(seq)
        dp = np.zeros_like(bpp)
        dp_diag_i = np.zeros_like(bpp)
        dp_diag_j = np.zeros_like(bpp)
        tr = np.zeros(bpp.shape, dtype=np.int)
        allowed_bp = self.allowed_basepairs(seq, allowed_bp)
        s = (gamma + 1) * bpp - 1
        if margin is not None:
            s += margin

        for l in range(1, N):
            dp_diag_1 = np.diag(dp, k=l-1)
            v_1 = dp_diag_1[1:].reshape((N-l, 1))
            v_2 = dp_diag_1[:-1].reshape((N-l, 1))
            if l >= 2:
                dp_diag_2 = np.diag(dp, k=l-2)
                v_3 = dp_diag_2[1:-1] + np.diag(s, k=l)
                v_3[np.diag(allowed_bp, k=l)==False] = -1e10
                v_3 = v_3.reshape((N-l), 1)
            else:
                v_3 = np.full((N-l,1), -1e10, dtype=bpp.dtype)
            v_k = dp_diag_i[:-l, :l] + dp_diag_j[l:, :l][:, ::-1]

            v = np.hstack((v_3, v_2, v_1, v_k))
            dp_diag = v.max(axis=1)
            tr_diag = v.argmax(axis=1)-3

            dp += np.diag(dp_diag, k=l) 
            tr += np.diag(tr_diag, k=l)
            dp_diag_i[:-l, l] = dp_diag
            dp_diag_j[l:, l] = dp_diag

        return dp, tr


    def traceback(self, tr, i, j, pair):
        if i < j:
            if tr[i, j] == -1:
                pair = self.traceback(tr, i+1, j, pair)
            elif tr[i, j] == -2:
                pair = self.traceback(tr, i, j-1, pair)
            elif tr[i, j] == -3:
                pair.append((i,j))
                pair = self.traceback(tr, i+1, j-1, pair)
            elif tr[i, j] >= 0:
                k = tr[i, j]+i
                pair = self.traceback(tr, i, k, pair)
                pair = self.traceback(tr, k+1, j, pair)
        return pair

    def calc_score(self, seq, bpp, pair, gamma=None, margin=None):
        gamma = self.gamma if gamma is None else gamma
        s = np.zeros((1,1), dtype=np.float32)
        if isinstance(bpp, cp.ndarray):
            s = cp.asarray(s)
        if isinstance(bpp, Variable):
            s = bpp.xp.zeros((1,1), dtype=np.float32)
            s = Variable(s)

        for i, j in pair:
            s += (gamma + 1) * bpp[i, j] - 1
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
    
    nussinov = Nussinov(gamma=9.0)
    for fa in read_fasta(sys.argv[1]):
        rna = RNA.fold_compound(fa['seq'])
        rna.pf()
        print(">"+fa['name'])
        print(fa['seq'])
        bpp = np.array(rna.bpp())
        pred = nussinov.decode(fa['seq'], bpp[1:, 1:])
        print(nussinov.dot_parenthesis(fa['seq'], pred))
