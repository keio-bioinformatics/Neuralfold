import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .base import BaseModel
from .bpmatrix import BatchedBPMatrix, BPMatrix


class CNN(BaseModel):
    hyper_params = ('width', 'layers', 'channels', 'dropout_rate', 'no_resnet', 'use_bn', 'use_dilate', 'hidden_nodes')
    params_prefix = 'cnn'

    def __init__(self, layers=1, channels=32, width=16, 
            dropout_rate=None, use_dilate=False, use_bn=False, 
            no_resnet=False, hidden_nodes=128):
        super(CNN, self).__init__()

        self.layers = layers
        self.out_channels = channels * 2
        self.width = width * 2 + 1
        self.dropout_rate = dropout_rate
        self.resnet = not no_resnet
        self.hidden_nodes = hidden_nodes
        self.use_bn = use_bn
        self.use_dilate = use_dilate

        self.config = {
            '--learning-model': 'CNN',
            '--cnn-width': width,
            '--cnn-layers': layers,
            '--cnn-channels': channels,
            '--cnn-dropout-rate': dropout_rate,
            '--cnn-no-resnet': no_resnet,
            '--cnn-use-bn': use_bn,
            '--cnn-use-dilate': use_dilate,
            '--cnn-hidden-nodes': hidden_nodes
        }

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


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for CNN model')
        group.add_argument('--cnn-width',
                        help='filter width (default: 5)',
                        type=int, default=5)
        group.add_argument('--cnn-layers',
                        help='the number of layers (default: 4)',
                        type=int, default=4)
        group.add_argument('--cnn-channels',
                        help='the number of output channels (default: 32)',
                        type=int, default=32)
        group.add_argument('--cnn-dropout-rate',
                        help='Dropout rate (default: 0.0)',
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
                        help='the number of hidden nodes in the fc layer (default: 128)',
                        type=int, default=128)


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
            if self.use_bn:
                x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
            else:
                x = F.leaky_relu(self.fc1(x))
            y = F.sigmoid(self.fc2(x))
            y = y.reshape(B, N-k) 
            bpp_diags.append(y)

        return BatchedBPMatrix(bpp_diags)
