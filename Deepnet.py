import numpy as np
import Config
#import Recursive
import chainer.links as L
import chainer.functions as F
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

class Deepnet(Chain):

    def __init__(self, hidden1, hidden2, hidden3, activation_function):
        # define model

        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.activation_function = activation_function


        super(Deepnet, self).__init__(
            L1 = L.Linear(None, self.hidden1),
            L2 = L.Linear(None , self.hidden2),
            L3_1 = L.Linear(None, 1),
            L3_2 = L.Linear(None, 2),

            L1_left = L.Linear(None, self.hidden1),
            L2_left = L.Linear(None , self.hidden2),
            L3_1_left = L.Linear(None, 1),
            L3_2_left = L.Linear(None, 2),

            L1_right = L.Linear(None, self.hidden1),
            L2_right = L.Linear(None , self.hidden2),
            L3_1_right = L.Linear(None, 1),
            L3_2_right = L.Linear(None, 2),
        )

    def __call__(self, x, direction):
        # データを受け取った際のforward計算を書く
        if direction == "center":
            h = F.leaky_relu(self.L1(x))
            if self.hidden2:
                h = F.leaky_relu(self.L2(h))

            if self.activation_function == "softmax":
                h = F.softmax(self.L3_2(h))
            elif self.activation_function == "sigmoid":
                h = F.sigmoid(self.L3_1(h))
            else:
                print("enexpected function")

        elif direction == "left":
            h = F.leaky_relu(self.L1_left(x))
            if self.hidden2:
                h = F.leaky_relu(self.L2_left(h))

            if self.activation_function == "softmax":
                h = F.softmax(self.L3_2_left(h))
            elif self.activation_function == "sigmoid":
                h = F.sigmoid(self.L3_1_left(h))
            else:
                print("enexpected function")

        elif direction == "right":
            h = F.leaky_relu(self.L1_right(x))
            if self.hidden2:
                h = F.leaky_relu(self.L2_right(h))
            if self.activation_function == "softmax":
                h = F.softmax(self.L3_2_right(h))
            elif self.activation_function == "sigmoid":
                h = F.sigmoid(self.L3_1_right(h))
            else:
                print("enexpected function")
        else:
            print("enexpected direction")
        return h
