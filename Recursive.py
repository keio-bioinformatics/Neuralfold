import numpy as np
import Config
#import Recursive
import chainer.links as L
import chainer.functions as F
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

feature_length = Config.feature_length
hidden = Config.hidden

class Recursive_net(Chain):

    def __init__(self):
        # パラメータを持つ層の登録
        super(Recursive_net, self).__init__(
            l1 = L.Linear(None , hidden),
            l2 = L.Linear(hidden , feature_length),
            L1 = L.Linear(None , hidden),
            L2 = L.Linear(hidden , 1)
        )

    def __call__(self, x , inner = True):
        # データを受け取った際のforward計算を書く
        if inner:
            h = F.relu(self.l1(x))
            h = F.relu(self.l2(h))

        else:
            h = F.relu(self.L1(x))
            h = F.sigmoid(self.L2(h))
        return h
