import numpy as np
import Config
#import Recursive
import chainer.links as L
import chainer.functions as F
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

class Deepnet(Chain):

    # def __init__(self,args):
    #     # define model
    #     self.hidden_insideoutside = args.hidden_insideoutside
    #     self.hidden2_insideoutside = args.hidden2_insideoutside
    #
    #     self.feature = args.feature
    #
    #     self.hidden_marge = args.hidden_marge
    #     self.hidden2_marge = args.hidden2_marge
    #     self.activation_function = args.activation_function

    def __init__(self,hidden_insideoutside,hidden2_insideoutside,feature, hidden_marge,hidden2_marge, activation_function):
        # define model

        self.hidden_marge = hidden_marge
        self.hidden2_marge = hidden2_marge
        self.activation_function = activation_function


        super(Deepnet, self).__init__(
            L1 = L.Linear(None, self.hidden_marge),
            L2 = L.Linear(None , self.hidden2_marge),
            L3_1 = L.Linear(None, 1),
            L3_2 = L.Linear(None, 2),
        )

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h = F.leaky_relu(self.L1(x))
        if self.hidden2_marge:
            h = F.leaky_relu(self.L2(h))
            if self.activation_function == "softmax":
                h = F.softmax(self.L3_2(h))
            elif self.activation_function == "sigmoid":
                h = F.sigmoid(self.L3_1(h))
            else:
                print("enexpected function")

        else:
            if self.activation_function == "softmax":
                h = F.softmax(self.L3_2(h))
            elif self.activation_function == "sigmoid":
                h = F.sigmoid(self.L3_1(h))
            else:
                print("enexpected function")
        return h
