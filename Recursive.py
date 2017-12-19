import numpy as np
import Config
#import Recursive
import chainer.links as L
import chainer.functions as F
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

class Recursive_net(Chain):

    def __init__(self,hidden = 80,feature = 80):
        # パラメータを持つ層の登録
        super(Recursive_net, self).__init__(
            # l1 = L.Linear(feature_length * 3 + base_length * 2, hidden),
            l1 = L.Linear(None, hidden),
            # l2 = L.Linear(hidden, hidden),
            l3 = L.Linear(hidden , feature),
            # L1 = L.Linear(feature_length * 2 , hidden),
            L1 = L.Linear(None, hidden),
            L2 = L.Linear(hidden , hidden),
            L3 = L.Linear(hidden , 1)
        )

    def __call__(self, x , inner = True):
        # データを受け取った際のforward計算を書く
        if inner:
            h = F.relu(self.l1(x))
            # h = F.relu(self.l2(h))
            h = F.relu(self.l3(h))

        else:
            h = F.relu(self.L1(x))
            h = F.relu(self.L2(h))
            h = F.sigmoid(self.L3(h))
        return h
