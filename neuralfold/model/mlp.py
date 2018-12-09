import math
import pickle

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .util import base_represent


class MLP(Chain):

    def __init__(self, neighbor=None, hidden1=None, hidden2=None, mlp_dropout_rate=None):
        self.neighbor = neighbor
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout_rate = mlp_dropout_rate

        super(MLP, self).__init__()
        with self.init_scope():
            self.L1 = L.Linear(None, self.hidden1)
            self.L2 = L.Linear(None , self.hidden2)
            self.L3_1 = L.Linear(None, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.L1(x))
        if self.dropout_rate is not None:
            h = F.dropout(h, ratio=self.dropout_rate)
        if self.hidden2:
            h = F.leaky_relu(self.L2(h))
            if self.dropout_rate is not None:
                h = F.dropout(h, ratio=self.dropout_rate)
        h = F.sigmoid(self.L3_1(h))
        # h = F.leaky_relu(self.L3_1(h))
        return h


    def save_model(self, f):
        with open(f+'.pickle', 'wb') as fp:
            pickle.dump(self.__class__.__name__, fp)
            pickle.dump(self.parse_args(self), fp)
        serializers.save_npz(f+'.npz', self)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for MLP model')
        group.add_argument('-hn1','--hidden1',
                        help='hidden layer nodes for neighbor model',
                        type=int, default=200)
        group.add_argument('-hn2','--hidden2',
                        help='hidden layer nodes2 for neighbor model',
                        type=int, default=50)
        group.add_argument('-n','--neighbor',
                        help='length of neighbor bases to see',
                        type=int, default=40)
        group.add_argument('--mlp-dropout-rate',
                        help='Dropout rate',
                        type=float)


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('neighbor', 'hidden1', 'hidden2', 'mlp_dropout_rate')
        return {p: getattr(args, p, None) for p in hyper_params if getattr(args, p, None) is not None}


    def base_onehot(self, base):
        if base in ['A' ,'a']:
            return np.array([1,0,0,0,0] , dtype=np.float32)
        elif base in ['U' ,'u']:
            return np.array([0,1,0,0,0] , dtype=np.float32)
        elif base in ['G' ,'g']:
            return np.array([0,0,1,0,0] , dtype=np.float32)
        elif base in ['C' ,'c']:
            return np.array([0,0,0,1,0] , dtype=np.float32)
        else:
            return np.array([0,0,0,0,0] , dtype=np.float32)


    def make_onehot_vector(self, seq):
        B = len(seq) # batch size
        M = 5 # the number of bases + 1
        N = max([len(s) for s in seq]) # the maximum length of sequences
        seq_vec = np.zeros((B, N, M), dtype=np.float32)
        for k, s in enumerate(seq):
            for i, base in enumerate(s):
                seq_vec[k, i, :] = self.base_onehot(base)
        return seq_vec


    def make_context_vector(self, seq_vec):
        B, N, M = seq_vec.shape # (batchsize, seqlen, bases)
        W = self.neighbor # the width of context with bases
        K = M*(2*W+1) # the width of both context with onehot
        zero_vec = np.zeros((B, W, M), dtype=np.float32)
        seq_vec = np.concatenate((zero_vec, seq_vec, zero_vec), axis=1)
        ret = np.zeros((B, N, K), dtype=np.float32)
        for i in range(N):
            ret[:, i, :] = seq_vec[:, i:i+W*2+1, :].reshape(B, K)
        return ret


    def make_interval_vector(self, interval, bit_len, scale):
        i = int(math.log(interval, scale)+1.)
        return np.eye(bit_len, dtype=np.float32)[min(i, bit_len-1)]


    def make_input_vector(self, v, interval, bit_len=20, scale=1.5):
        B, N, K = v.shape
        W = self.neighbor

        v_l = v[:, :-interval, :] # (B, N-interval, K)
        v_r = v[:, interval:, :]  # (B, N-interval, K)
        v_int = self.make_interval_vector(interval, bit_len, scale) # (bit_len,)
        v_int = np.tile(v_int, (B, N-interval, 1)) # (B, N-interval, bit_len)
        x = np.concatenate((v_l, v_r, v_int), axis=2) # (B, N-interval, K*2+bit_len)
        if interval < W:
            l = W - interval
            x[:, :, K-l*5:K+l*5] = np.tile([0, 0, 0, 0, 1], l*2).astype(np.float32)
        return x


    def make_input_vector_ij(self, v, i, j, bit_len=20, scale=1.5):
        B, _, K = v.shape
        W = self.neighbor
        interval = j-i
        v_int = self.make_interval_vector(interval, bit_len, scale) # (bit_len,)
        v_int = np.tile(v_int, (B, 1)) # (B, bit_len)
        x = np.concatenate((v[:, i, :], v[:, j, :], v_int), axis=1) # (B, K*2+bit_len)

        if interval < W:
            l = W - interval
            x[:, K-l*5:K+l*5] = np.tile([0, 0, 0, 0, 1], l*2).astype(np.float32)
        return x


    def compute_bpp_0(self, seq):
        seq_vec = self.make_onehot_vector(seq)
        B, N, _ = seq_vec.shape
        seq_vec = self.make_context_vector(seq_vec)

        bpp = Variable(np.empty((B, 0, N), dtype=np.float32))
        for i in range(N):
            bpp_i = Variable(np.empty((B, 1, 0), dtype=np.float32))
            for j in range(N):
                if i<j:
                    x = self.make_input_vector_ij(seq_vec, i, j)    
                    y_ij = self(x).reshape(B, 1, 1)
                else:  
                    y_ij = Variable(np.zeros((B, 1, 1), dtype=np.float32))
                bpp_i = F.concat((bpp_i, y_ij), axis=2)
            bpp = F.concat((bpp, bpp_i), axis=1)
        # print(bpp.shape) : (B, N, N)
        return bpp


    def diagonalize(self, x, k=0):
        xp = self.xp
        B, N_orig = x.shape
        N = N_orig + abs(k)
        if k>0:
            x = F.hstack((xp.zeros((B, k), dtype=x.dtype), x))
        elif k<0:
            x = F.hstack((x, xp.zeros((B, -k), dtype=x.dtype)))
        x = F.tile(x, N).reshape(B, N, N)
        cond = xp.diag(xp.ones(N_orig, dtype=np.bool), k=k)
        return F.where(cond, x, xp.zeros((N, N), dtype=np.float32))


    def compute_bpp(self, seq):
        xp = self.xp
        seq_vec = self.make_onehot_vector(seq)
        B, N, _ = seq_vec.shape
        seq_vec = self.make_context_vector(seq_vec)

        bpp = Variable(xp.zeros((B, N, N), dtype=np.float32))
        for k in range(1, N):
            x = self.make_input_vector(seq_vec, k) # (B, N-k, *)
            x = x.reshape(B*(N-k), -1)
            x = xp.asarray(x)
            y = self(x) # (B*(N-k), 1)
            y = y.reshape(B, N-k) 
            y = self.diagonalize(y, k=k)
            cond = np.diag(np.ones((N-k,), dtype=np.bool), k=k)
            bpp = F.where(cond, y, bpp)

        return bpp