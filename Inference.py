import numpy as np
import Config
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers

min_loop_length = Config.min_loop_length
FEATURE_SIZE = Config.feature_length
bifurcation = Config.bifurcation

class Inference:
    def __init__(self, seq):
        self.seq=seq
        self.N=len(self.seq)

    def pair_check(self,tup):
        if tup in [('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'),('G','U'),('U','G')]:
            return True
        return False

    def base_represent(self,base):
        if base in ['A' ,'a']:
            return Variable(np.array([[1,0,0,0]] , dtype=np.float32))
        elif base in ['U' ,'u']:
            return Variable(np.array([[0,1,0,0]] , dtype=np.float32))
        elif base in ['G' ,'g']:
            return Variable(np.array([[0,0,1,0]] , dtype=np.float32))
        elif base in ['C' ,'c']:
            return Variable(np.array([[0,0,0,1]] , dtype=np.float32))
        else:
            print("error")
            return Variable(np.array([[0,0,0,0]] , dtype=np.float32))

    def ComputeInsideOutside(self, model):

        BP = [[0 for i in range(self.N)] for j in range(self.N)]

        #define feature matrix
        FM_inside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        FM_outside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]


        #compute inside
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                x = F.concat((FM_inside[i][j-1] , FM_inside[i+1][j-1] , FM_inside[i+1][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                #if bifurcation:


                FM_inside[i][j] = model(x)

        #compute outside
        for n in range(self.N-1 , 1-1 , -1):
            for j in range(self.N-1 , n-1 , -1):
                i = j-n
                a = i-1
                b = j+1
                if i == 0:
                    a = self.N-1
                if j == self.N-1:
                    b = 0
                #print(i,j,a,b)
                x = F.concat((FM_outside[i][b] , FM_outside[a][b] , FM_outside[a][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                FM_outside[i][j] = model(x)

        #marge inside outside
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                x = F.concat((FM_inside[i][j] , FM_outside[i][j]) ,axis=1)
                BP[i][j] = model(x , inner = False)

        return BP

    def buildDP(self, BP):
        DP = np.zeros((self.N,self.N))
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                if self.pair_check((self.seq[i].upper(), self.seq[j].upper())):
                    case1 = DP[i+1,j-1] + BP[i][j].data
                else:
                    case1 = -100
                case2 = DP[i+1,j]
                case3 = DP[i,j-1]
                if i+3<=j:
                    tmp=np.array([])
                    for k in range(i+1,j):
                        tmp = np.append(tmp , DP[i,k] + DP[k+1,j])
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
            elif DP[i, j] == DP[i+1, j-1]+BP[i][j].data:
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
        pair = self.traceback(DP, BP, 0, self.N-1, np.empty((0,2),dtype=np.int16) )
        return pair
