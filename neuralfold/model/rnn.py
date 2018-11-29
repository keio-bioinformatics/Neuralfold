import pickle

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers


class RNN(Chain):
    def __init__(self, hidden_insideoutside, hidden2_insideoutside, feature, hidden_merge, hidden2_merge):
        self.hidden_insideoutside = hidden_insideoutside
        self.hidden2_insideoutside = hidden2_insideoutside
        self.feature = feature
        self.hidden_merge = hidden_merge
        self.hidden2_merge = hidden2_merge

        super(RNN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, self.hidden_insideoutside)
            self.l2 = L.Linear(None, self.hidden2_insideoutside)
            self.l3 = L.Linear(None, self.feature)
            self.L1 = L.Linear(None, self.hidden_merge)
            self.L2 = L.Linear(None , self.hidden2_merge)
            self.L3_1 = L.Linear(None, 1)
            self.L3_2 = L.Linear(None, 2)


    def __call__(self, x , inner = True):
        if inner:
            h = F.leaky_relu(self.l1(x))
            if self.hidden2_insideoutside:
                h = F.leaky_relu(self.l2(h))
            h = F.leaky_relu(self.l3(h))

        else:
            h = F.leaky_relu(self.L1(x))
            if self.hidden2_merge:
                h = F.leaky_relu(self.L2(h))
                h = F.sigmoid(self.L3_1(h))

            else:
                h = F.sigmoid(self.L3_1(h))
        return h

    def compute_inside(self, seq):
        xp = self.xp
        B, N, _ = seq.shape
        M = self.n_featues

        dp_in = Variable(xp.zeros((B, N, N, M), dtype=np.float32))
        dp_diag_i = Variable(xp.zeros_like(dp_in)) # (B, N, N, M)
        dp_diag_j = Variable(xp.zeros_like(dp_in)) # (B, N, N, M)
        for l in range(1, N):
            dp_diag_1 = F.diagonal(dp_in, offset=l-1, axis1=1, axis2=2)
            v_1 = dp_diag_1[:, 1:].reshape(B, N-l, M)
            v_2 = dp_diag_1[:, :-1].reshape(B, N-l, M)
            if l >= 2:
                dp_diag_2 = F.diagonal(dp_in, offset=l-2, axis1=1, axis2=2)
                v_3 = dp_diag_2[:, 1:-1].reshape(B, N-l, M)
            else:
                v_3 = Variable(xp.zeros((B, N-l, M), dtype=dp_in.dtype))
            v_k = dp_diag_i[:, :-l, :l, :] + dp_diag_j[:, l:, :l, :][:, :, ::-1, :] # (B, N-l, l, M)
            #v_k = F.mean(v_k, axis=2) # (B, N-l, M)
            v_k = F.logsumexp(v_k, axis=2) # (B, N-l, M)

            x_i = seq[:, l:, :] # (B, N-l, 4)
            x_j = seq[:, :-l, :] # (B, N-l, 4)
            v = F.hstack((x_i, x_j, v_1, v_2, v_3, v_k))
            dp_in_diag = self.forward_inside(v) # expect (B, N-l, M)

            # F.diag(dp_in.diag, k=l) = dp_in_diag
            dp_in += self.diagonalize(dp_in_diag, k=l) # (B, N, N, M)
            # dp_diag_i[:, :-l, l, :] = dp_in_diag
            # dp_diag_j[:, l:, l, :] = dp_in_diag
            dp_in_diag.reshape(B, N-l, 1, M)
            dp_diag_i += F.pad(dp_in_diag, ((0, 0), (0, l), (l, N-l-1), (0, 0)), mode='constant')
            dp_diag_j += F.pad(dp_in_diag, ((0, 0), (l, 0), (l, N-l-1), (0, 0)), mode='constant')

        return dp_in


    def compute_outside(self, seq, dp_in):
        xp = self.xp
        B, N, _ = seq.shape
        M = self.n_featues

        dp_out = Variable(xp.zeros((B, N, N, M), dtype=np.float32))
        dp_diag_i = Variable(xp.zeros_like(dp_out)) # (B, N, N, M)
        dp_diag_j = Variable(xp.zeros_like(dp_out)) # (B, N, N, M)

        # initialize

        for l in reversed(range(0, N-1)):
            dp_diag_1 = F.diagonal(dp_out, offset=l+1, axis1=1, axis2=2) # (B, N-(l+1), M)
            v_1 = F.pad(dp_diag_1, ((0, 0), (0, 1), (0, 0)), mode='constant') # (B, N-l, M)
            v_2 = F.pad(dp_diag_1, ((0, 0), (1, 0), (0, 0)), mode='constant') # (B, N-l, M)
            if l+2 < N:
                dp_diag_2 = F.diagonal(dp_out, offset=l+2, axis1=1, axis2=2) # (B, N-(l+2), M)
                v_3 = F.pad(dp_diag_2, ((0, 0), (1, 1), (0, 0)), mode='constant') # (B, N-l, M)
            else:
                v_3 = Variable(xp.zeros((B, N-l, M), dtype=dp_out.dtype))
            v_k_i = dp_diag_i[:, :-l, l, :] # (B, N-l, 1, M)
            v_k_j = dp_diag_j[:, l:, l, :] # (B, N-l, 1, M)
            v_k = F.logsumexp(F.concat((v_k_i, v_k_j), axis=2), axis=2) # (B, N-l, M)

            x_i = seq[:, l:, :] # (B, N-l, 4)
            x_j = seq[:, :-l, :] # (B, N-l, 4)
            v = F.hstack((x_i, x_j, v_1, v_2, v_3, v_k))
            dp_out_diag = self.forward_outside(v) # expect (B, N-l, M)

            dp_out += self.diagonal(dp_out_diag, k=l) # (B, N, N, M)
            # invalid
            dp_diag_i[:, :-l, :l, :] ^= dp_out_diag + dp_diag_in_j[:, l:, :l, :][:, :, ::-1, :] # logsumexp
            dp_diag_j[:, l:, :l, :][:, :, ::-1, :] ^= dp_out_diag + dp_diag_in_i[:, :-1, :l, :] # logsumexp

        return dp_in


    def save_model(self, f):
        with open(f+'.pickle', 'wb') as fp:
            pickle.dump(self.__class__.__name__, fp)
            pickle.dump(self.parse_args(self), fp)
        serializers.save_npz(f+'.npz', self)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for RNN model')
        group.add_argument('-H1','--hidden_insideoutside',
                            help = 'hidden layer nodes for inside outside',
                            type=int, default=80)
        group.add_argument('-H2','--hidden2_insideoutside',
                            help = 'hidden layer nodes2 for inside outside',
                            type=int)
        group.add_argument('-h1','--hidden_merge',
                            help = 'hidden layer nodes for merge phase',
                            type=int, default=80)
        group.add_argument('-h2','--hidden2_merge',
                            help = 'hidden layer nodes2 for merge phase',
                            type=int)
        group.add_argument('-f','--feature',
                            help = 'feature length',
                            type=int, default=40)


    @classmethod
    def parse_args(cls, args):
        params = ('hidden_insideoutside', 'hidden2_insideoutside', 
                'hidden_merge', 'hidden2_merge', 'feature')
        return {p: getattr(args, p, None) for p in params if getattr(args, p, None) is not None}
