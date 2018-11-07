import pickle
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import Chain, serializers


class RNN(Chain):
    def __init__(self, hidden_insideoutside, hidden2_insideoutside, feature, hidden_merge, hidden2_merge):
        self.hidden_insideoutside = hidden_insideoutside
        self.hidden2_insideoutside = hidden2_insideoutside
        self.feature = feature
        self.hidden_merge = hidden_merge
        self.hidden2_merge = hidden2_merge

        super(RNN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, self.hidden_insideoutside)
            self.l2 = L.Linear(None, self.hidden2_insideoutside)
            self.l3 = L.Linear(None, self.feature)
            self.L1 = L.Linear(None, self.hidden_merge)
            self.L2 = L.Linear(None , self.hidden2_merge)
            self.L3_1 = L.Linear(None, 1)
            self.L3_2 = L.Linear(None, 2)


    def __call__(self, x , inner = True):
        if inner:
            h = F.leaky_relu(self.l1(x))
            if self.hidden2_insideoutside:
                h = F.leaky_relu(self.l2(h))
            h = F.leaky_relu(self.l3(h))

        else:
            h = F.leaky_relu(self.L1(x))
            if self.hidden2_merge:
                h = F.leaky_relu(self.L2(h))
                h = F.sigmoid(self.L3_1(h))

            else:
                h = F.sigmoid(self.L3_1(h))
        return h


    def save_model(self, f):
        with open(f+'.pickle', 'wb') as fp:
            pickle.dump(self.__class__.__name__, fp)
            pickle.dump(self.parse_args(self), fp)
        serializers.save_npz(f+'.npz', self)


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for RNN model')
        group.add_argument('-H1','--hidden_insideoutside',
                            help = 'hidden layer nodes for inside outside',
                            type=int, default=80)
        group.add_argument('-H2','--hidden2_insideoutside',
                            help = 'hidden layer nodes2 for inside outside',
                            type=int)
        group.add_argument('-h1','--hidden_merge',
                            help = 'hidden layer nodes for merge phase',
                            type=int, default=80)
        group.add_argument('-h2','--hidden2_merge',
                            help = 'hidden layer nodes2 for merge phase',
                            type=int)
        group.add_argument('-f','--feature',
                            help = 'feature length',
                            type=int, default=40)


    @classmethod
    def parse_args(cls, args):
        params = ('hidden_insideoutside', 'hidden2_insideoutside', 
                'hidden_merge', 'hidden2_merge', 'feature')
        return {p: getattr(args, p, None) for p in params if getattr(args, p, None) is not None}

