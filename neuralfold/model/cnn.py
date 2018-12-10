import math
import pickle

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .util import base_represent


class CNN(Chain):

    def __init__(self, layers, channels, width, 
            dropout_rate=None, use_dilate=False, use_bn=False, 
            no_resnet=False, hidden_nodes=128):
        super(CNN, self).__init__()

        self.layers = layers
        self.out_channels = channels
        self.width = width
        self.dropout_rate = dropout_rate
        self.resnet = not no_resnet
        self.hidden_nodes = hidden_nodes
        self.use_bn = use_bn
        for i in range(self.layers):
            if use_dilate:
                conv = L.Convolution1D(None, self.out_channels, self.width,
                                        dilate=2**i, pad=2**i*(self.width//2))
            else:
                conv = L.Convolution1D(None, self.out_channels, self.width, pad=self.width//2)
            self.add_link("conv{}".format(i), conv)
            if self.use_bn:
                self.add_link("bn{}".format(i), L.BatchNormalization(self.out_channels))
        with self.init_scope():
            self.fc1 = L.Linear(None, self.hidden_nodes)
            if self.use_bn:
                self.bn_fc1 = L.BatchNormalization(self.hidden_nodes)
            self.fc2 = L.Linear(None, 1)


    def forward(self, x):
        h = F.swapaxes(x, 1, 2)
        for i in range(self.layers):
            h1 = h
            h = self['conv{}'.format(i)](h)
            if self.use_bn:
                h = self['bn{}'.format(i)](h)
            h = F.leaky_relu(h)
            if self.dropout_rate is not None:
                h = F.dropout(h, ratio=self.dropout_rate)
            if self.resnet and h.shape == h1.shape:
                h += h1
        return F.swapaxes(h, 1, 2)


    def save_model(self, f):
        with open(f+'.pickle', 'wb') as fp:
            pickle.dump(self.__class__.__name__, fp)
            pickle.dump(self.parse_args(self), fp)
        serializers.save_npz(f+'.npz', self)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for CNN model')
        group.add_argument('--cnn-width',
                        help='filter width',
                        type=int, default=5)
        group.add_argument('--cnn-layers',
                        help='the number of layers',
                        type=int, default=4)
        group.add_argument('--cnn-channels',
                        help='the number of output channels',
                        type=int, default=32)
        group.add_argument('--cnn-dropout-rate',
                        help='Dropout rate',
                        type=float)
        group.add_argument('--cnn-no-resnet',
                        help='disallow use of residual connections',
                        action='store_true')
        group.add_argument('--cnn-use-bn',
                        help='use batch normalization',
                        action='store_true')
        group.add_argument('--cnn-use-dilate',
                        help='use dilated convolutional networks',
                        action='store_true')
        group.add_argument('--cnn-hidden-nodes',
                        help='the number of hidden nodes in the fc layer',
                        type=int, default=128)


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('width', 'layers', 'channels', 'dropout_rate', 'no_resnet', 'use_bn', 'use_dilate', 'hidden_nodes')
        return {p: getattr(args, 'cnn_'+p, None) for p in hyper_params if getattr(args, 'cnn_'+p, None) is not None}


    def base_onehot(self, base):
        if base in ['A' ,'a']:
            return np.array([1,0,0,0] , dtype=np.float32)
        elif base in ['U' ,'u']:
            return np.array([0,1,0,0] , dtype=np.float32)
        elif base in ['G' ,'g']:
            return np.array([0,0,1,0] , dtype=np.float32)
        elif base in ['C' ,'c']:
            return np.array([0,0,0,1] , dtype=np.float32)
        else:
            return np.array([0,0,0,0] , dtype=np.float32)


    def make_onehot_vector(self, seq):
        B = len(seq) # batch size
        M = 4 # the number of bases + 1
        N = max([len(s) for s in seq]) # the maximum length of sequences
        seq_vec = np.zeros((B, N, M), dtype=np.float32)
        for k, s in enumerate(seq):
            for i, base in enumerate(s):
                seq_vec[k, i, :] = self.base_onehot(base)
        return seq_vec


    def make_interval_vector(self, interval, bit_len, scale):
        i = int(math.log(interval, scale)+1.)
        return self.xp.eye(bit_len, dtype=np.float32)[min(i, bit_len-1)]


    def make_input_vector(self, v, interval, bit_len=20, scale=1.5):
        B, N, _ = v.shape
        v_l = v[:, :-interval, :] # (B, N-interval, K)
        v_l = F.split_axis(v_l, 2, axis=2)[0]
        v_r = v[:, interval:, :]  # (B, N-interval, K)
        v_r = F.split_axis(v_r, 2, axis=2)[1]
        v_int = self.make_interval_vector(interval, bit_len, scale) # (bit_len,)
        v_int = self.xp.tile(v_int, (B, N-interval, 1)) # (B, N-interval, bit_len)
        x = F.concat((v_l+v_r, v_int), axis=2) # (B, N-interval, K/2+bit_len)
        return x


    def make_input_vector_0(self, v, interval, bit_len=20, scale=1.5):
        B, N, _ = v.shape
        v_l = v[:, :-interval, :] # (B, N-interval, K)
        v_r = v[:, interval:, :]  # (B, N-interval, K)
        v_int = self.make_interval_vector(interval, bit_len, scale) # (bit_len,)
        v_int = self.xp.tile(v_int, (B, N-interval, 1)) # (B, N-interval, bit_len)
        x = F.concat((v_l, v_r, v_int), axis=2) # (B, N-interval, K*2+bit_len)
        return x


    def make_input_vector_ij(self, v, i, j, bit_len=20, scale=1.5):
        B, N, _ = v.shape
        interval = j - i
        v_l = v[:, :-interval, :] # (B, N-interval, K)
        v_r = v[:, interval:, :]  # (B, N-interval, K)
        v_int = self.make_interval_vector(interval, bit_len, scale) # (bit_len,)
        v_int = self.xp.tile(v_int, (B, N-interval, 1)) # (B, N-interval, bit_len)
        x = F.concat((v_l, v_r, v_int), axis=2) # (B, N-interval, K*2+bit_len)
        return x[:, i, :]


    def compute_bpp_0(self, seq):
        xp = self.xp
        seq_vec = self.make_onehot_vector(seq) # (B, N, 4)
        B, N, _ = seq_vec.shape
        seq_vec = xp.asarray(seq_vec)
        seq_vec = self.forward(seq_vec) # (B, N, out_ch)

        bpp = Variable(np.empty((B, 0, N), dtype=np.float32))
        for i in range(N):
            bpp_i = Variable(np.empty((B, 1, 0), dtype=np.float32))
            for j in range(N):
                if i<j:
                    x = self.make_input_vector_ij(seq_vec, i, j)    
                    x = F.leaky_relu(self.fc1(x))
                    y_ij = F.sigmoid(self.fc2(x))
                    y_ij = y_ij.reshape(B, 1, 1) 
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

    #@profile
    def compute_bpp(self, seq):
        xp = self.xp
        seq_vec = self.make_onehot_vector(seq) # (B, N, 4)
        B, N, _ = seq_vec.shape
        seq_vec = xp.asarray(seq_vec)
        seq_vec = self.forward(seq_vec) # (B, N, out_ch)
        allowed_bp = self.xp.asarray(self.allowed_basepairs(seq))

        bpp = Variable(xp.zeros((B, N, N), dtype=np.float32))
        for k in range(1, N):
            x = self.make_input_vector(seq_vec, k) # (B, N-k, *)
            x = x.reshape(B*(N-k), -1)
            if self.use_bn:
                x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
            else:
                x = F.leaky_relu(self.fc1(x))
            y = F.sigmoid(self.fc2(x))
            y = y.reshape(B, N-k) 
            y = self.diagonalize(y, k=k)
            cond = xp.diag(xp.ones(N-k, dtype=np.bool), k=k)
            bpp = F.where(cond & allowed_bp, y, bpp)

        return bpp


    def allowed_basepairs(self, seq, allowed_bp='canonical'):
        B = len(seq)
        N = max([len(s) for s in seq])

        if allowed_bp is None:
            return self.xp.ones((B, N, N), dtype=np.bool)

        elif isinstance(allowed_bp, str) and allowed_bp == 'canonical':
            allowed_bp = np.zeros((B, N, N), dtype=np.bool)
            canonicals = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}
            for k, s in enumerate(seq):
                s = s.upper()
                for j in range(len(s)):
                    for i in range(j):
                        if (s[i], s[j]) in canonicals and j-i>2:
                            allowed_bp[k, i, j] = allowed_bp[k, j, i] = True
            return allowed_bp

        elif isinstance(allowed_bp, list):
            a = np.zeros((B, N, N), dtype=np.bool)
            for k, bp in enumerate(allowed_bp):
                for i, j in bp:
                    a[k, i, j] = a[k, j, i] = True
            return a
