import numpy as np
import Config
#import Recursive
import chainer.links as L
import chainer.functions as F
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

min_loop_length = Config.min_loop_length
FEATURE_SIZE = Config.feature_length


class Recursive_net(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        # パラメータを持つ層の登録
        super(Recursive_net, self).__init__(
            l1 = L.Linear(None , feature_length),
            l2 = L.Linear(None , 1)
        )

    def __call__(self, x , inner = True):
        # データを受け取った際のforward計算を書く
        if(inner):
            h = F.relu(self.l1(x))
        else:
            h = F.relu(self.l2(x))
        return h


class Inference:
    def __init__(self, seq):
        self.seq=seq
        self.N=len(self.seq)
    #initialize matrix with zeros where can't have pairings
    def initializeBP(self):
        #NxN matrix that stores the BaseParing probability
        #     BP = np.empty((self.N,self.N))
        #     BP[:] = np.NAN
        #BP = np.zeros((self.N,self.N))
        BP = np.random.rand(self.N,self.N)
        return BP


    #initialize Feature matrix
    #def initializeFM(self):

    def pair_check(self,tup):
        if tup in [('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'),('G','U'),('U','G')]:
            return True
        return False

    def base_represent(self,base):
        if base == 'A'


    def ComputeInsideOutside(self, model):

        #BP = self.initializeBP()
        BP = np.zeros((self.N,self.N))


        #define feature matrix
        FM_inside = Variable(np.zeros((self.N , self.N , FEATURE_SIZE)) , dtype=np.float32)
        FM_outside = Variable(np.zeros((self.N , self.N , FEATURE_SIZE)), dtype=np.float32)
        zero_vector = Variable(np.zeros((1,FEATURE_SIZE)), dtype=np.float32)
        #define model
        # model = Recursive_net()
        # optimizer = chainer.optimizer
        # optimizer.setup(model)

        #compute inside
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                x = F.concat((FM_inside[i,j-1] , FM_inside[i+1,j-1] , FM_inside[i+1,j] , base_represent(self.seq[i]) , base_represent(self.seq[j])) ,axis=1)
                FM_inside[i,j] = model(x)

        #compute outside
        for n in range(self.N-1 , 1-1 , -1):
            for j in range(self.N-1 , n-1 , -1):
                i = j-n
                if i == 0 and j ==　self.N-1:
                    x = F.concat((zero_vector , zero_vector , zero_vector , base_represent(self.seq[i]) , base_represent(self.seq[j])) ,axis=1)
                elif i ==0:
                    x = F.concat((FM_outside[i,j+1] , zero_vector , zero_vector , base_represent(self.seq[i]) , base_represent(self.seq[j])) ,axis=1)
                elif j == self.N-1:
                    x = F.concat((zero_vector , zero_vector , FM_outside[i-1,j] , base_represent(self.seq[i]) , base_represent(self.seq[j])) ,axis=1)
                else:
                    x = F.concat((FM_outside[i,j+1] , FM_outside[i-1,j+1] , FM_outside[i-1,j] , base_represent(self.seq[i]) , base_represent(self.seq[j])) ,axis=1)

                FM_outside[i,j] = model(x)

        #marge inside outside
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                x = F.concat((FM_inside[i , j] , FM_outside[i , j]) ,axis=1)
                BP[i,j] = model(x , inner = False)

        return BP


    def buildDP(self, BP):
        DP = np.zeros((self.N,self.N))
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                if self.pair_check((self.seq[i], self.seq[j])):
                    #print('A')
                    case1 = DP[i+1,j-1]+BP[i,j]
                else:
                    case1 = -100
                case2 = DP[i+1,j]
                case3 = DP[i,j-1]
                if i+3<=j:
                    tmp=np.array([])
                    for k in range(i+1,j):
                        tmp = np.append(tmp , DP[i,k]+DP[k+1,j])
                    case4 = np.max(tmp)
                    DP[i,j] = max(case1,case2,case3,case4)
                else:
                    DP[i,j] = max(case1,case2,case3)
        return DP

    def traceback(self, DP, BP, i, j, pair):
        if i < j:
            if DP[i, j] == DP[i+1, j]:
                pair = self.traceback(DP, BP, i+1, j, pair)
            elif DP[i, j] == DP[i, j-1]:
                pair = self.traceback(DP, BP, i, j-1, pair)
            elif DP[i, j] == DP[i+1, j-1]+BP[i,j]:
                pair = np.append(pair, [[i,j]], axis=0)
                pair = self.traceback(DP, BP, i+1, j-1, pair)
            else:
                for k in range(i+1,j):
                    if DP[i,j] == DP[i,k]+DP[k+1,j]:
                        pair = self.traceback(DP, BP, i, k, pair)
                        pair = self.traceback(DP, BP, k+1, j, pair)
                        break
        return pair


    def ComputePosterior(self, BP):
        DP = self.buildDP(BP)
        pair = self.traceback(DP, BP, 0, self.N-1, np.empty((0,2)) )
        return pair
