import math
import pickle

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .util import base_represent


class MLP(Chain):

    def __init__(self, neighbor=None, hidden1=None, hidden2=None):
        self.neighbor = neighbor
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        super(MLP, self).__init__()
        with self.init_scope():
            self.L1 = L.Linear(None, self.hidden1)
            self.L2 = L.Linear(None , self.hidden2)
            self.L3_1 = L.Linear(None, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.L1(x))
        if self.hidden2:
            h = F.leaky_relu(self.L2(h))
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
                        help = 'hidden layer nodes for neighbor model',
                        type=int, default=200)
        group.add_argument('-hn2','--hidden2',
                        help = 'hidden layer nodes2 for neighbor model',
                        type=int, default=50)
        group.add_argument('-n','--neighbor',
                        help = 'length of neighbor bases to see',
                        type=int, default=40)


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('neighbor', 'hidden1', 'hidden2')
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


    def compute_bpp_1(self, seq):
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


    def compute_bpp(self, seq):
        seq_vec = self.make_onehot_vector(seq)
        B, N, _ = seq_vec.shape
        seq_vec = self.make_context_vector(seq_vec)

        bpp = [ 
            [ Variable(self.xp.zeros((B, 1, 1), dtype=np.float32)) for _ in range(N) ] for _ in range(N)
        ]
        for k in range(1, N):
            x = self.make_input_vector(seq_vec, k) # (B, N-k, *)
            x = x.reshape(B*(N-k), -1)
            x = self.xp.asarray(x)
            y = self(x).reshape(B, N-k)
            for i in range(N-k):
                j = i + k
                bpp[i][j] = y[:, i].reshape(-1, 1, 1)

        bpp = [ F.concat(bpp_i, axis=2) for bpp_i in bpp ]
        bpp = F.concat(bpp, axis=1)

        return bpp


    def compute_bpp_0(self, seq):
        seq = seq[0]
        N = len(seq)
        base_length = 5
        neighbor = self.neighbor
        sequence_vector = np.empty((0, base_length), dtype=np.float32)
        for base in seq:
            sequence_vector = np.vstack((sequence_vector, base_represent(base)))
        sequence_vector = Variable(sequence_vector)
        # sequence_vector.shape: (N, base_length)

        # Fill the side with 0 vectors
        sequence_vector_neighbor = sequence_vector.reshape(N, base_length) # ??
        sequence_vector_neighbor = F.vstack((sequence_vector_neighbor, Variable(np.zeros((neighbor, base_length), dtype=np.float32)) ))
        sequence_vector_neighbor = F.vstack((Variable(np.zeros((neighbor, base_length), dtype=np.float32)) , sequence_vector_neighbor))
        # sequence_vector_neighbor.shape: (N+neighbor*2, base_length)

        # Create vectors around each base
        side_input_size =  base_length * (neighbor * 2 + 1)
        side_input_vector = Variable(np.empty((0, side_input_size), dtype=np.float32))
        for base in range(0, N):
            side_input_vector = F.vstack((side_input_vector, sequence_vector_neighbor[base:base+(neighbor * 2 + 1)].reshape(1,side_input_size)))
        # side_input_vector.shape: (N, side_input_size=base_length*(neighbor*2+1))

        # initiate BPP table
        BP = Variable(np.zeros((N, 1, 1), dtype=np.float32))

        # predict BPP
        for interval in range(1, N):
            left = side_input_vector[0 : N - interval]
            right = side_input_vector[interval : N]
            x = F.hstack((left, right))
            # x.shape: (N-interval, 2*side_input_size)

            # If there is an overlap, fill that portion with a specific vector
            if interval <= neighbor:
                row = N-interval
                column = (neighbor-interval+1)*2*base_length
                a = int(column/base_length)
                surplus = Variable(np.array(([0,0,0,0,1] * row * a), dtype=np.float32))
                surplus = surplus.reshape(row, column)
                x[0:row,side_input_size-int(column/2)+1:side_input_size+int(column/2)+1].data = surplus.data

            # add location information
            direct = False
            onehot = True
            if direct:
                left_location = Variable(np.arange(1, N - interval+1, dtype=np.float32).reshape(N - interval,1))/10
                right_location = Variable(np.arange(interval+1, N+1, dtype=np.float32).reshape(N - interval,1))/10
                full_length = Variable(np.full(N - interval, N, dtype=np.float32).reshape(N - interval,1))/10
                x = F.hstack((x, left_location, right_location, full_length))
            elif onehot:
                # left_location = Variable(np.eye(80)[np.ceil(np.arange(1, N - interval+1, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                # right_location = Variable(np.eye(80)[np.ceil(np.arange(interval+1, N+1, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                # full_length = Variable(np.eye(80)[np.ceil(np.full(N - interval, N, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                interval_length = Variable(np.eye(20)[np.ceil(np.full(N - interval, math.log(interval,1.5), dtype=np.float32)).astype(np.int8)].astype(np.float32))
                # x = F.hstack((x, left_location, right_location, full_length))
                x = F.hstack((x, interval_length))
                # x.shape(N-interval, 2*side_input_size+20)

            BP = F.vstack((BP, self(x).reshape(N - interval,1,1)))

        # reshape to (N,N)
        BP = F.flatten(BP)
        diagmat = Variable(np.empty((0, N), dtype=np.float32))
        i = 0
        for l in range(N, 0, -1):
            x1 = BP[i:i+l].reshape(1, l)
            x2 = np.zeros((1, N-l), dtype=np.float32)
            x = F.hstack((x2, x1))
            diagmat = F.vstack((diagmat, x))
            i += l

        bpp = Variable(np.empty((N, 0), dtype=np.float32))
        for i in range(N):
            x = F.hstack((diagmat[:i+1, i][::-1], diagmat[i+1:, i]))
            bpp = F.hstack((bpp, x.reshape(N, 1)))

        return bpp
