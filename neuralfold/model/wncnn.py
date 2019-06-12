import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .base import BaseModel
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
        h, g = F.split_axis(h, 2, axis=1) # Gated Convolution
        h = F.dropout(h * F.sigmoid(g), self.do_rate)
        return h


class WNCNN(BaseModel):
    hyper_params = ('targets', 'width', 'layers', 'channels', 'dropout_rate', 'hidden_nodes')
    params_prefix = 'wncnn'


    def __init__(self, layers, channels, width, 
            dropout_rate=None, targets=5, hidden_nodes=128):
        super(WNCNN, self).__init__()

        self.layers = layers
        self.out_channels = channels * 2
        self.width = width * 2 + 1
        self.dropout_rate = dropout_rate
        self.n_targets = targets * 2
        self.hidden_nodes = hidden_nodes
        self.bit_len = 8
        self.scale = 2
        in_ch = 4

        self.config = {
            '--learning-model': 'WNCNN',
            '--wncnn-targets': targets,
            '--wncnn-width': width,
            '--wncnn-layers': layers,
            '--wncnn-channels': channels,
            '--wncnn-dropout-rate': dropout_rate,
            '--wncnn-hidden-nodes': hidden_nodes
        }

        for i in range(self.layers):
            conv = DilatedBlock(in_ch, self.out_channels, self.width, 
                                dilate=2**i, do_rate=self.dropout_rate)
            self.add_link("conv{}".format(i), conv)
            in_ch += self.out_channels
        with self.init_scope():
            self.l = L.ConvolutionND(1, None, self.n_targets, 1)
            self.fc1 = WNLinear(self.n_targets // 2 + self.bit_len, self.hidden_nodes)
            self.fc2 = WNLinear(self.hidden_nodes, 1)


    def forward(self, x):
        h = F.swapaxes(x, 1, 2)
        hs = [h]
        for i in range(self.layers):
            h = self['conv{}'.format(i)](hs)
            hs.append(h)
        h = self.l(F.concat(hs, axis=1))
        return F.swapaxes(h, 1, 2)


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


    #@profile
    def compute_bpp(self, seq):
        seq_vec = self.make_onehot_vector(seq) # (B, N, 4)
        B, N, _ = seq_vec.shape
        seq_vec = self.xp.asarray(seq_vec)
        seq_vec = self.forward(seq_vec) # (B, N, out_ch)

        bpp_diags = [None]
        for k in range(1, N):
            x = self.make_input_vector(seq_vec, k, self.bit_len, self.scale) # (B, N-k, *)
            x = x.reshape(B*(N-k), -1)
            x = F.leaky_relu(self.fc1(x))
            y = F.sigmoid(self.fc2(x))
            y = y.reshape(B, N-k) 
            bpp_diags.append(y)

        return BatchedBPMatrix(bpp_diags)
