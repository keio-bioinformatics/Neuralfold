import chainer.links as L
import chainer.functions as F
from chainer import Chain

class Deepnet(Chain):

    def __init__(self, neighbor, hidden1, hidden2):
        self.neighbor = neighbor
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        super(Deepnet, self).__init__(
            L1 = L.Linear(None, self.hidden1),
            L2 = L.Linear(None , self.hidden2),
            L3_1 = L.Linear(None, 1),
        )


    def __call__(self, x):
        h = F.leaky_relu(self.L1(x))
        if self.hidden2:
            h = F.leaky_relu(self.L2(h))
        h = F.sigmoid(self.L3_1(h))
        # h = F.leaky_relu(self.L3_1(h))
        return h
