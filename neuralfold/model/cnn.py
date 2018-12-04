import math
import pickle

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .util import base_represent


class CNN(Chain):

    def __init__(self, layers, channels, width, dropout_rate=None, no_resnet=False):
        super(CNN, self).__init__()

        self.layers = layers
        self.out_channels = channels
        self.width = width
        self.dropout_rate = dropout_rate
        self.resnet = not no_resnet

        for i in range(self.layers):
            self.add_link("conv{}".format(i), L.Convolution1D(None, self.out_channels, self.width, pad=self.width//2))
        with self.init_scope():
            self.fc = L.Linear(None, 1)


    def forward(self, x):
        h = F.swapaxes(x, 1, 2)
        for i in range(self.layers):
            h1 = h
            h = self['conv{}'.format(i)](h)
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


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('width', 'layers', 'channels', 'dropout_rate', 'no_resnet')
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
        v_r = v[:, interval:, :]  # (B, N-interval, K)
        v_int = self.make_interval_vector(interval, bit_len, scale) # (bit_len,)
        v_int = self.xp.tile(v_int, (B, N-interval, 1)) # (B, N-interval, bit_len)
        x = F.concat((v_l, v_r, v_int), axis=2) # (B, N-interval, K*2+bit_len)
        return x


    def diagonalize(self, x, k=0):
        xp = self.xp
        B, N_orig = x.shape
        N = N_orig + abs(k)
        if k>0:
            x = F.hstack((xp.zeros((B, k), dtype=x.dtype), x))
        elif k<0:
            x = F.hstack((x, xp.zeros((B, -k), dtype=x.dtype)))
        x = F.tile(x, N).reshape(B, N, N)
        return x * xp.diag(xp.ones(N_orig), k=k)


    def compute_bpp(self, seq):
        xp = self.xp
        seq_vec = self.make_onehot_vector(seq) # (B, N, 4)
        B, N, _ = seq_vec.shape
        seq_vec = xp.asarray(seq_vec)
        seq_vec = self.forward(seq_vec) # (B, N, out_ch)

        bpp = Variable(xp.zeros((B, N, N), dtype=np.float32))
        for k in range(1, N):
            x = self.make_input_vector(seq_vec, k) # (B, N-k, *)
            x = x.reshape(B*(N-k), -1)
            y = F.sigmoid(self.fc(x))
            y = y.reshape(B, N-k) 
            bpp += self.diagonalize(y, k=k) # (B, N, N)

        return bpp
