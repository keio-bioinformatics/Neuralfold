import numpy as np
import Config
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
from joblib import Parallel, delayed

min_loop_length = Config.min_loop_length
FEATURE_SIZE = Config.feature_length
bifurcation = Config.bifurcation
PARALLEL = Config.Parallel
gpu = Config.gpu

class Inference:
    def __init__(self, seq, model):
        self.seq=seq
        self.N=len(self.seq)
        self.FM_inside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        self.FM_outside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        self.model =model

    def pair_check(self,tup):
        if tup in [('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'),('G','U'),('U','G')]:
            return True
        return False

    def base_represent(self,base):
        if gpu == True:
            print("gpu")
            # xp = cuda.cupy
            # if base in ['A' ,'a']:
            #     return Variable(xp.array([[1,0,0,0]] , dtype=np.float32))
            # elif base in ['U' ,'u']:
            #     return Variable(xp.array([[0,1,0,0]] , dtype=np.float32))
            # elif base in ['G' ,'g']:
            #     return Variable(xp.array([[0,0,1,0]] , dtype=np.float32))
            # elif base in ['C' ,'c']:
            #     return Variable(xp.array([[0,0,0,1]] , dtype=np.float32))
            # else:
            #     print("error")
            #     return Variable(xp.array([[0,0,0,0]] , dtype=np.float32))
        else:
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

    def ComputeInside_Parallel(self, i, j):
        # print("inside"+str(i)+str(j))
        x = F.concat((self.FM_inside[i][j-1] , self.FM_inside[i+1][j-1] , self.FM_inside[i+1][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
        return self.model(x)

    def ComputeOutside_Parallel(self, i, j):
        # print("outside"+str(i)+str(j))
        a = i-1
        b = j+1
        if i == 0:
            a = self.N-1
        if j == self.N-1:
            b = 0
        x = F.concat((self.FM_outside[i][b] , self.FM_outside[a][b] , self.FM_outside[a][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
        return self.model(x)

    def ComputeInsideOutside(self):

        BP = [[0 for i in range(self.N)] for j in range(self.N)]


        # gpu_device = 0
        # cuda.get_device(gpu_device).use()
        # model.to_gpu(gpu_device)
        # xp = cuda.cupy


        #define feature matrix
        #use gpu
        # if gpu == True:
        #     print("gpu")
        #     # self.FM_inside = [[Variable(xp.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        #     # self.FM_outside = [[Variable(xp.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        # #use cpu
        # else:
        #     self.FM_inside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        #     self.FM_outside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]


        #compute inside
        for n in range(1,self.N):
            #Parallel
            if PARALLEL == True and self.N-n > 0:
                r =  Parallel(n_jobs=-1)([delayed(self.ComputeInside_Parallel)(j-n, j) for j in range(n,self.N)])
                for j in range(n,self.N):
                    self.FM_inside[j-n][j] = r.pop(0)


            else:
                for j in range(n,self.N):
                    i = j-n
                    x = F.concat((self.FM_inside[i][j-1] , self.FM_inside[i+1][j-1] , self.FM_inside[i+1][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                    self.FM_inside[i][j] = self.model(x)

        #compute outside
        for n in range(self.N-1 , 1-1 , -1):
            if PARALLEL == True and self.N-n > 0:
                # print("here1")
                r =  Parallel(n_jobs=-1)([delayed(self.ComputeOutside_Parallel)(j-n, j) for j in range(self.N-1 , n-1 , -1)])
                for j in range(self.N-1 , n-1 , -1):
                    self.FM_outside[j-n][j] = r.pop(0)

            else:
                for j in range(self.N-1 , n-1 , -1):
                    i = j-n
                    a = i-1
                    b = j+1
                    if i == 0:
                        a = self.N-1
                    if j == self.N-1:
                        b = 0
                    #print(i,j,a,b)
                    x = F.concat((self.FM_outside[i][b] , self.FM_outside[a][b] , self.FM_outside[a][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                    self.FM_outside[i][j] = self.model(x)

        #marge inside outside
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                x = F.concat((self.FM_inside[i][j] , self.FM_outside[i][j]) ,axis=1)
                BP[i][j] = self.model(x , inner = False)
        print(BP)
        return BP

    def buildDP(self, BP):
        DP = np.zeros((self.N,self.N))
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                if self.pair_check((self.seq[i].upper(), self.seq[j].upper())):
                    case1 = DP[i+1,j-1] + BP[i][j].data
                else:
                    case1 = -1000
                case2 = DP[i+1,j]
                case3 = DP[i,j-1]
                if i+3<=j:
                    tmp = np.array([])
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
