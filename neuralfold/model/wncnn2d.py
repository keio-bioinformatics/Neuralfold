import math
import pickle

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .util import base_represent
from .bpmatrix import BatchedBPMatrix, BPMatrix

class WNLinear(L.Linear):
    def __init__(self, *args, **kwargs):
        super(WNLinear, self).__init__(*args, **kwargs)
        self.add_param('g', self.W.data.shape[0])
        norm = np.linalg.norm(self.W.data, axis=1)
        self.g.data[...] = norm # pylint: disable=E1101

    def __call__(self, x):
        norm = F.batch_l2_norm_squared(self.W) ** 0.5
        norm_broadcasted = F.broadcast_to(
            F.expand_dims(norm, 1), self.W.data.shape)
        g_broadcasted = F.broadcast_to(
            F.expand_dims(self.g, 1), self.W.data.shape) # pylint: disable=E1101
        return F.linear(x, g_broadcasted * self.W / norm_broadcasted, self.b)


class WNConvolutionND(L.ConvolutionND):
    def __init__(self, *args, **kwargs):
        super(WNConvolutionND, self).__init__(*args, **kwargs)
        self.add_param('g', self.W.data.shape[0])
        norm = np.linalg.norm(self.W.data.reshape(
            self.W.data.shape[0], -1), axis=1)
        self.g.data[...] = norm # pylint: disable=E1101

    def __call__(self, x):
        norm = F.batch_l2_norm_squared(self.W) ** 0.5
        channel_size = self.W.data.shape[0]
        norm_broadcasted = F.broadcast_to(
            F.reshape(norm, (channel_size, 1, 1, 1)), self.W.data.shape)
        g_broadcasted = F.broadcast_to(
            F.reshape(norm, (channel_size, 1, 1, 1)), self.W.data.shape)
        return F.convolution_nd(
            x, g_broadcasted * self.W / norm_broadcasted, self.b, self.stride,
            self.pad, self.cover_all, self.dilate)


class DilatedBlock(Chain):
    def __init__(self, in_ch, out_ch, kernel, dilate, do_rate, ndim=1):
        super(DilatedBlock, self).__init__()
        self.do_rate = do_rate
        with self.init_scope():
            self.conv = WNConvolutionND(ndim, in_ch, out_ch*2, kernel, pad=dilate, dilate=dilate)

    def forward(self, xs):
        x = F.concat(xs, axis=1)
        h = self.conv(x)
        h, g = F.split_axis(h, 2, axis=1) # Gated Convolution
        h = F.dropout(h * F.sigmoid(g), self.do_rate)
        return h


class WNCNN2D(Chain):

    def __init__(self, layers, channels, kernel, 
            dropout_rate=None, targets=5, hidden_nodes=128):
        super(WNCNN2D, self).__init__()

        self.layers = layers
        self.out_channels = channels * 2
        self.kernel = kernel * 2 + 1
        self.dropout_rate = dropout_rate
        self.n_targets = targets * 2
        self.hidden_nodes = hidden_nodes
        self.bit_len = 8
        self.scale = 2
        in_ch = 4*2+self.bit_len
        for i in range(self.layers):
            conv = DilatedBlock(in_ch, self.out_channels, self.kernel, 
                                dilate=2**i, do_rate=self.dropout_rate, ndim=2)
            self.add_link("conv{}".format(i), conv)
            in_ch += self.out_channels
        with self.init_scope():
            self.l = L.ConvolutionND(2, None, self.n_targets, 1)
            self.fc1 = WNLinear(self.n_targets, self.hidden_nodes)
            self.fc2 = WNLinear(self.hidden_nodes, 1)


    def forward(self, x):
        return self.compute_bpp(x)


    def save_model(self, f):
        with open(f+'.pickle', 'wb') as fp:
            pickle.dump(self.__class__.__name__, fp)
            pickle.dump(self.parse_args(self), fp)
        serializers.save_npz(f+'.npz', self)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for WNCNN2D model')
        group.add_argument('--wncnn2d-targets',
                        help='n_targets',
                        type=int, default=10)
        group.add_argument('--wncnn2d-kernel',
                        help='filter kernel',
                        type=int, default=1)
        group.add_argument('--wncnn2d-layers',
                        help='the number of layers',
                        type=int, default=4)
        group.add_argument('--wncnn2d-channels',
                        help='the number of output channels',
                        type=int, default=24)
        group.add_argument('--wncnn2d-dropout-rate',
                        help='Dropout rate',
                        type=float, default=0.01)
        group.add_argument('--wncnn2d-hidden-nodes',
                        help='the number of hidden nodes in the fc layer',
                        type=int, default=128)


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('targets', 'kernel', 'layers', 'channels', 'dropout_rate', 'hidden_nodes')
        return {p: getattr(args, 'wncnn2d_'+p, None) for p in hyper_params if getattr(args, 'wncnn2d_'+p, None) is not None}


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
        M = 4 # the number of bases
        N = max([len(s) for s in seq]) # the maximum length of sequences
        seq_vec = np.zeros((B, N, M), dtype=np.float32)
        for k, s in enumerate(seq):
            for i, base in enumerate(s):
                seq_vec[k, i, :] = self.base_onehot(base)
        return seq_vec


    def make_interval_vector(self, interval, bit_len, scale):
        i = int(math.log(interval, scale)+1.) if interval > 0 else 0
        return np.eye(bit_len, dtype=np.float32)[min(i, bit_len-1)]


    def make_input_tensor(self, seq, bit_len=8, scale=2):
        v = self.make_onehot_vector(seq)
        B, N, K = v.shape
        int_v = [ self.make_interval_vector(interval, bit_len, scale) for interval in range(N) ]
        int_v = np.broadcast_to(int_v, (B, N, bit_len))
        x = np.zeros((B, N, N, K*2+bit_len), dtype=np.float32)
        for j in range(N):
            for i in range(j):
                if bit_len > 0:
                    iv = int_v[:, j-i, :]
                    x[:, i, j, :] = np.concatenate((v[:, i, :], v[:, j, :], iv), axis=1)
                else:
                    x[:, i, j, :] = np.concatenate((v[:, i, :], v[:, j, :]), axis=1)
        return x


    def compute_bpp(self, seq):
        x = self.make_input_tensor(seq, self.bit_len, self.scale) # (B, N, N, K+bit_len)
        B, N, _, _ = x.shape
        x = self.xp.asarray(x)
        x = F.transpose(x, (0, 3, 1, 2)) # (B, K+bit_len, N, N)
        hs = [x]
        for i in range(self.layers):
            h = self['conv{}'.format(i)](hs)
            hs.append(h)
        h = self.l(F.concat(hs, axis=1))
        h = F.transpose(h, (0, 2, 3, 1)) # (B, N, N, n_target)
        h = h.reshape(B*N*N, -1)
        h = F.leaky_relu(self.fc1(h))
        y = F.sigmoid(self.fc2(h))
        y = y.reshape(B, N, N)
        return y
