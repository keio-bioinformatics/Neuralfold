import numpy as np
import Config
#import Recursive
import chainer.links as L
import chainer.functions as F
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

class Recursive_net(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        # パラメータを持つ層の登録
        super(Recursive_net, self).__init__(
            l1 = L.Linear(None , feature_length),
            l2 = L.Linear(None , 1)
        )

    def __call__(self, x , inner = True):
        # データを受け取った際のforward計算を書く
        if inner:
            h = F.relu(self.l1(x))
        else:
            h = F.sigmoid(self.l2(x))
        return h
