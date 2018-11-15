import numpy as np
from chainer import Variable

from . import Decoder

class Nussinov(Decoder):
    def __init__(self, gamma=None, use_simple_dp=False):
        self.gamma = gamma
        self.use_simple_dp = use_simple_dp

    def decode(self, bpp, gamma=None, margin=None, allowed_bp=None):
        '''Nussinov-style decoding algorithm    
        '''
        gamma = self.gamma if gamma is None else gamma
        N = len(bpp)
        bpp = bpp.data if type(bpp) is Variable else bpp
        if self.use_simple_dp:
            _, tr = self.build_dp(bpp, gamma=gamma, allowed_bp=allowed_bp, margin=margin)
        else:
            _, tr = self.build_dp_v(bpp, gamma=gamma, allowed_bp=allowed_bp, margin=margin)
        pair = self.traceback(tr, 0, N-1, [])
        return pair

    def build_dp(self, bpp, gamma, margin=None, allowed_bp=None):
        N = len(bpp)
        dp = np.zeros(bpp.shape, dtype=np.float32)
        tr = np.zeros(bpp.shape, dtype=np.int)

        for j in range(N):
            for i in reversed(range(j)):
                s = (gamma + 1) *  bpp[i, j] - 1
                if margin is not None:
                    s += margin[i, j]
                if i+1 < j:
                    dp[i, j] = dp[i+1, j]
                    tr[i, j] = 1
                if i < j-1 and dp[i, j] < dp[i, j-1]:
                    dp[i, j] = dp[i, j-1]
                    tr[i, j] = 2
                if i+1 < j-1 and dp[i, j] < dp[i+1, j-1] + s:
                    dp[i, j] = dp[i+1, j-1] + s
                    tr[i, j] = 3
                for k in range(i+1, j):
                    if dp[i, j] < dp[i, k] + dp[k+1, j]:
                        dp[i, j] = dp[i, k] + dp[k+1, j]
                        tr[i, j] = k-i+3
        
        return dp, tr

    #@profile
    def build_dp_v(self, bpp, gamma, margin=None, allowed_bp=None):
        N = len(bpp)
        dp = np.zeros_like(bpp)
        dp_diag_i = np.zeros_like(bpp)
        dp_diag_j = np.zeros_like(bpp)
        tr = np.zeros_like(bpp, dtype=np.int)
        s = (gamma + 1) * bpp - 1
        if margin is not None:
            s += margin

        for k in range(1, N):
            dp_diag_1 = np.diag(dp, k=k-1)
            v_1 = dp_diag_1[1:].reshape((N-k, 1))
            v_2 = dp_diag_1[:-1].reshape((N-k, 1))
            if k >= 2:
                dp_diag_2 = np.diag(dp, k=k-2)
                v_3 = dp_diag_2[1:-1] + np.diag(s, k=k)
                v_3 = v_3.reshape((N-k), 1)
            else:
                v_3 = np.zeros((N-k,1), dtype=bpp.dtype)
            v_k = dp_diag_i[:-k, :k] + dp_diag_j[k:, :k][:, ::-1]

            v = np.hstack((v_1, v_2, v_3, v_k))
            dp_diag = v.max(axis=1)
            tr_diag = v.argmax(axis=1)+1

            dp += np.diag(dp_diag, k=k) 
            tr += np.diag(tr_diag, k=k)
            dp_diag_i[:-k, k] = dp_diag
            dp_diag_j[k:, k] = dp_diag

        return dp, tr

    def traceback(self, tr, i, j, pair):
        if i < j:
            if tr[i, j] == 1:
                pair = self.traceback(tr, i+1, j, pair)
            elif tr[i, j] == 2:
                pair = self.traceback(tr, i, j-1, pair)
            elif tr[i, j] == 3:
                pair.append((i,j))
                pair = self.traceback(tr, i+1, j-1, pair)
            elif tr[i, j] > 3:
                k = i+tr[i, j]-3
                pair = self.traceback(tr, i, k, pair)
                pair = self.traceback(tr, k+1, j, pair)
        return pair

    def calc_score(self, bpp, pair, gamma=None, margin=None):
        gamma = self.gamma if gamma is None else gamma
        s = np.zeros((1,1), dtype=np.float32)
        if type(bpp) is Variable:
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
        pred = nussinov.decode(bpp[1:, 1:])
        print(nussinov.dot_parenthesis(fa['seq'], pred))
