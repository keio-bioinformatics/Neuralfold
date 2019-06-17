import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, Variable, serializers

from .base import BaseModel
from .bpmatrix import BatchedBPMatrix, BPMatrix


class ResNet(BaseModel):
    hyper_params = ('ksize1d', 'blocks1d', 'targets1d', 
        'ksize2d', 'blocks2d', 'targets2d', 'dropout_rate', 'hidden_nodes')
    params_prefix = 'resnet'

    def __init__(self, args=None, ksize1d=17, blocks1d=3, targets1d=32,
        ksize2d=3, blocks2d=32, targets2d=32, dropout_rate=0.0, hidden_nodes=32):
        super(ResNet, self).__init__()

        self.config['--learning-model'] =  self.__class__.__name__
        for n in self.hyper_params:
            v = locals()[n] if args is None else getattr(args, self.params_prefix+'_'+n, None)
            setattr(self, n, v)
            self.config['--'+self.params_prefix+'-'+n.replace('_', '-')] = v

        with self.init_scope():
            # pylint: disable=maybe-no-member
            self.conv1d_0 = L.Convolution1D(None, self.targets1d, self.ksize1d, pad=self.ksize1d//2)
            for i in range(1, self.blocks1d+1):
                for j in (1, 2):
                    self.add_link("bn1d_{}_{}".format(i, j), 
                        L.BatchNormalization(self.targets1d))
                    self.add_link("conv1d_{}_{}".format(i, j), 
                        L.Convolution1D(None, self.targets1d, self.ksize1d, pad=self.ksize1d//2))
            self.bn1d_last = L.BatchNormalization(self.targets1d)
            
            self.conv2d_0 = L.Convolution2D(None, self.targets2d, self.ksize2d, pad=self.ksize2d//2)
            for i in range(1, self.blocks2d+1):
                for j in (1, 2):
                    self.add_link("bn2d_{}_{}".format(i, j), 
                        L.BatchNormalization(self.targets2d))
                    self.add_link("conv2d_{}_{}".format(i, j),
                        L.Convolution2D(None, self.targets2d, self.ksize2d, pad=self.ksize2d//2))
            self.bn2d_last = L.BatchNormalization(self.targets2d)

            self.fc1 = L.Linear(None, self.hidden_nodes)
            self.fc2 = L.Linear(None, 1)


    def forward1d(self, x):
        h = F.swapaxes(x, 1, 2) # (B, M, N)
        h = self.conv1d_0(h) # (B, targets1d, N)
        for i in range(1, self.blocks1d+1): # pylint: disable=maybe-no-member
            h1 = h
            for j in (1, 2):
                h = F.relu(h)
                h = self['bn1d_{}_{}'.format(i, j)](h)
                h = self['conv1d_{}_{}'.format(i, j)](h)
            h += h1 # (B, targets1d, N)
        h = F.relu(h)
        h = self.bn1d_last(h)
        return F.swapaxes(h, 1, 2) # (B, N, targets1d)


    def forward2d(self, x): # (B, N, N, targets1d*2)
        x = F.transpose(x, axes=(0, 3, 1, 2)) # (B, targets1d*2, N, N)
        h = self.conv2d_0(x) # (B, targets2d, N, N)
        for i in range(1, self.blocks2d+1): # pylint: disable=maybe-no-member
            h1 = h
            for j in (1, 2):
                h = F.relu(h)
                h = self['bn2d_{}_{}'.format(i, j)](h)
                h = self['conv2d_{}_{}'.format(i, j)](h)
            h += h1 # (B, targets2d, N, N)
        h = F.relu(h)
        h = self.bn2d_last(h)
        return F.transpose(h, axes=(0, 2, 3, 1)) # (B, N, N, targets2d)


    def forward(self, seq):
        self.compute_bpp(seq)


    #@profile
    def compute_bpp(self, seq):
        seq_vec = self.make_onehot_vector(seq) # (B, N, 4)
        B, N, _ = seq_vec.shape
        seq_vec = self.xp.asarray(seq_vec)
        seq_vec = self.forward1d(seq_vec) # (B, N, targets1d)

        seq_mat = F.tile(seq_vec, (1, 1, N)) # (B, N, N*targets1d)
        seq_mat = seq_mat.reshape(B, N, N, -1) # (B, N, N, targets1d)
        seq_mat_t = F.transpose(seq_mat, (0, 2, 1, 3))
        seq_mat = F.concat((seq_mat, seq_mat_t), axis=3) # (B, N, N, targets1d*2)
        cond = self.xp.asarray(np.fromfunction(lambda i, j, k, l: j < k, seq_mat.shape))
        seq_mat = F.where(cond, seq_mat, self.xp.zeros(seq_mat.shape, dtype=np.float32))
        seq_mat = self.forward2d(seq_mat) # (B, N, N, targets2d)

        bpp_diags = [None]
        for k in range(1, N):
            x = F.diagonal(seq_mat, offset=k, axis1=1, axis2=2) # (B, N-k, targets2d)
            x = x.reshape(B*(N-k), -1)
            h = F.relu(self.fc1(x)) # (B*(N-k), hidden_nodes)
            h = F.dropout(h, ratio=self.dropout_rate) # pylint: disable=maybe-no-member
            y = F.sigmoid(self.fc2(h))
            y = y.reshape(B, N-k) 
            bpp_diags.append(y)

        return BatchedBPMatrix(bpp_diags)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for ResNet model')
        group.add_argument('--resnet-ksize1d',
                        help='filter width (default: 17)',
                        type=int, default=17)
        group.add_argument('--resnet-blocks1d',
                        help='the number of 1D ResNet blocks (default: 3)',
                        type=int, default=3)
        group.add_argument('--resnet-targets1d',
                        help='the number of output channels of 1D ResNet blocks (default: 32)',
                        type=int, default=32)
        group.add_argument('--resnet-ksize2d',
                        help='filter width (default: 3)',
                        type=int, default=3)
        group.add_argument('--resnet-blocks2d',
                        help='the number of 1D ResNet blocks (default: 32)',
                        type=int, default=32)
        group.add_argument('--resnet-targets2d',
                        help='the number of output channels of 1D ResNet blocks (default: 32)',
                        type=int, default=32)
        group.add_argument('--resnet-dropout-rate',
                        help='Dropout rate (default: 0.5)',
                        type=float, default=0.5)
