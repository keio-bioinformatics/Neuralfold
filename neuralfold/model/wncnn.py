import math
import pickle

import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .util import base_represent
from .bpmatrix import BatchedBPMatrix, BPMatrix


class WNConvolutionND(L.ConvolutionND):
    def __init__(self, *args, **kwargs):
        super(WNConvolutionND, self).__init__(*args, **kwargs)
        self.add_param('g', self.W.data.shape[0])
        norm = np.linalg.norm(self.W.data.reshape(
            self.W.data.shape[0], -1), axis=1)
        self.g.data[...] = norm

    def __call__(self, x):
        norm = F.batch_l2_norm_squared(self.W) ** 0.5
        channel_size = self.W.data.shape[0]
        norm_broadcasted = F.broadcast_to(
            F.reshape(norm, (channel_size, 1, 1)), self.W.data.shape)
        g_broadcasted = F.broadcast_to(
            F.reshape(norm, (channel_size, 1, 1)), self.W.data.shape)
        return F.convolution_nd(
            x, g_broadcasted * self.W / norm_broadcasted, self.b, self.stride,
            self.pad, self.cover_all, self.dilate)


class DilatedBlock(Chain):
    def __init__(self, in_ch, out_ch, kernel, dilate, do_rate):
        super(DilatedBlock, self).__init__()
        self.do_rate = do_rate
        with self.init_scope():
            self.conv = WNConvolutionND(1, in_ch, out_ch*2, kernel, pad=dilate, dilate=dilate)

    def forward(self, xs):
        x = F.concat(xs, axis=1)
        h = self.conv(x)
        h, g = F.split_axis(h, 2, 1) # Gated Convolution
        h = F.dropout(h * F.sigmoid(g), self.do_rate)
        return h


class WNCNN(Chain):

    def __init__(self, layers, channels, width, 
            dropout_rate=None, targets=10, hidden_nodes=128):
        super(WNCNN, self).__init__()

        self.layers = layers
        self.out_channels = channels * 2
        self.width = width * 2 + 1
        self.dropout_rate = dropout_rate
        self.n_targets = targets
        self.hidden_nodes = hidden_nodes
        in_ch = 4
        for i in range(self.layers):
            conv = DilatedBlock(in_ch, self.out_channels, self.width, 
                                dilate=2**i, do_rate=self.dropout_rate)
            self.add_link("conv{}".format(i), conv)
            in_ch += self.out_channels
        with self.init_scope():
            self.l = L.ConvolutionND(1, None, self.n_targets, 1)
            self.fc1 = L.Linear(None, self.hidden_nodes)
            self.fc2 = L.Linear(None, 1)


    def forward(self, x):
        h = F.swapaxes(x, 1, 2)
        hs = [h]
        for i in range(self.layers):
            h = self['conv{}'.format(i)](hs)
            hs.append(h)
        self.l(F.concat(hs, axis=1))
        return F.swapaxes(h, 1, 2)


    def save_model(self, f):
        with open(f+'.pickle', 'wb') as fp:
            pickle.dump(self.__class__.__name__, fp)
            pickle.dump(self.parse_args(self), fp)
        serializers.save_npz(f+'.npz', self)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for CNN model')
        group.add_argument('--wncnn-targets',
                        help='n_targets',
                        type=int, default=10)
        group.add_argument('--wncnn-width',
                        help='filter width',
                        type=int, default=1)
        group.add_argument('--wncnn-layers',
                        help='the number of layers',
                        type=int, default=4)
        group.add_argument('--wncnn-channels',
                        help='the number of output channels',
                        type=int, default=24)
        group.add_argument('--wncnn-dropout-rate',
                        help='Dropout rate',
                        type=float, default=0.01)
        group.add_argument('--wncnn-hidden-nodes',
                        help='the number of hidden nodes in the fc layer',
                        type=int, default=128)


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('targets', 'width', 'layers', 'channels', 'dropout_rate', 'hidden_nodes')
        return {p: getattr(args, 'wncnn_'+p, None) for p in hyper_params if getattr(args, 'wncnn_'+p, None) is not None}


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


    #@profile
    def compute_bpp(self, seq):
        xp = self.xp
        seq_vec = self.make_onehot_vector(seq) # (B, N, 4)
        B, N, _ = seq_vec.shape
        seq_vec = xp.asarray(seq_vec)
        seq_vec = self.forward(seq_vec) # (B, N, out_ch)
        #allowed_bp = self.xp.asarray(self.allowed_basepairs(seq))

        bpp_diags = [None]
        for k in range(1, N):
            x = self.make_input_vector(seq_vec, k) # (B, N-k, *)
            x = x.reshape(B*(N-k), -1)
            x = F.leaky_relu(self.fc1(x))
            y = F.sigmoid(self.fc2(x))
            y = y.reshape(B, N-k) 
            bpp_diags.append(y)

        return BatchedBPMatrix(bpp_diags)


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
