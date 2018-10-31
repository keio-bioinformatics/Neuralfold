import math
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable
from .. import Config
from .util import base_represent

class MLP(Chain):

    def __init__(self, neighbor, hidden1, hidden2):
        self.neighbor = neighbor
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        super(MLP, self).__init__()
        with self.init_scope():
            self.L1 = L.Linear(None, self.hidden1)
            self.L2 = L.Linear(None , self.hidden2)
            self.L3_1 = L.Linear(None, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.L1(x))
        if self.hidden2:
            h = F.leaky_relu(self.L2(h))
        h = F.sigmoid(self.L3_1(h))
        # h = F.leaky_relu(self.L3_1(h))
        return h

    def compute_bpp(self, seq):
        N = len(seq)
        base_length = Config.base_length
        neighbor = self.neighbor
        sequence_vector = np.empty((0, base_length), dtype=np.float32)
        for base in seq:
            sequence_vector = np.vstack((sequence_vector, base_represent(base)))
        sequence_vector = Variable(sequence_vector)
        # sequence_vector.shape: (N, base_length)

        # Fill the side with 0 vectors
        sequence_vector_neighbor = sequence_vector.reshape(N, base_length) # ??
        sequence_vector_neighbor = F.vstack((sequence_vector_neighbor, Variable(np.zeros((neighbor, base_length), dtype=np.float32)) ))
        sequence_vector_neighbor = F.vstack((Variable(np.zeros((neighbor, base_length), dtype=np.float32)) , sequence_vector_neighbor))
        # sequence_vector_neighbor.shape: (N+neighbor*2, base_length)

        # Create vectors around each base
        side_input_size =  base_length * (neighbor * 2 + 1)
        side_input_vector = Variable(np.empty((0, side_input_size), dtype=np.float32))
        for base in range(0, N):
            side_input_vector = F.vstack((side_input_vector, sequence_vector_neighbor[base:base+(neighbor * 2 + 1)].reshape(1,side_input_size)))
        # side_input_vector.shape: (N, side_input_size=base_length*(neighbor*2+1))

        # initiate BPP table
        BP = Variable(np.zeros((N, 1, 1), dtype=np.float32))

        # predict BPP
        for interval in range(1, N):
            left = side_input_vector[0 : N - interval]
            right = side_input_vector[interval : N]
            x = F.hstack((left, right))
            # x.shape: (N-interval, 2*side_input_size)

            # If there is an overlap, fill that portion with a specific vector
            if interval <= neighbor:
                row = N-interval
                column = (neighbor-interval+1)*2*base_length
                a = int(column/base_length)
                surplus = Variable(np.array(([0,0,0,0,1] * row * a), dtype=np.float32))
                surplus = surplus.reshape(row, column)
                x[0:row,side_input_size-int(column/2)+1:side_input_size+int(column/2)+1].data = surplus.data

            # add location information
            direct = False
            onehot = True
            if direct:
                left_location = Variable(np.arange(1, N - interval+1, dtype=np.float32).reshape(N - interval,1))/10
                right_location = Variable(np.arange(interval+1, N+1, dtype=np.float32).reshape(N - interval,1))/10
                full_length = Variable(np.full(N - interval, N, dtype=np.float32).reshape(N - interval,1))/10
                x = F.hstack((x, left_location, right_location, full_length))
            elif onehot:
                # left_location = Variable(np.eye(80)[np.ceil(np.arange(1, N - interval+1, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                # right_location = Variable(np.eye(80)[np.ceil(np.arange(interval+1, N+1, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                # full_length = Variable(np.eye(80)[np.ceil(np.full(N - interval, N, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                interval_length = Variable(np.eye(20)[np.ceil(np.full(N - interval, math.log(interval,1.5), dtype=np.float32)).astype(np.int8)].astype(np.float32))
                # x = F.hstack((x, left_location, right_location, full_length))
                x = F.hstack((x, interval_length))
                # x.shape(N-interval, 2*side_input_size+20)

            BP = F.vstack((BP, self(x).reshape(N - interval,1,1)))

        # reshape to (N,N)
        BP = F.flatten(BP)
        diagmat = Variable(np.empty((0, N), dtype=np.float32))
        i = 0
        for l in range(N, 0, -1):
            x1 = BP[i:i+l].reshape(1, l)
            x2 = np.zeros((1, N-l), dtype=np.float32)
            x = F.hstack((x2, x1))
            diagmat = F.vstack((diagmat, x))
            i += l

        bpp = Variable(np.empty((N, 0), dtype=np.float32))
        for i in range(N):
            x = F.hstack((diagmat[:i+1, i][::-1], diagmat[i+1:, i]))
            bpp = F.hstack((bpp, x.reshape(N, 1)))

        return bpp