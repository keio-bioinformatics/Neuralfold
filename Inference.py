import numpy as np
import Config
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
from joblib import Parallel, delayed
from multiprocessing import Pool
import multiprocessing as multi
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import math

min_loop_length = Config.min_loop_length
FEATURE_SIZE = Config.feature_length
bifurcation = Config.bifurcation
PARALLEL = Config.Parallel
gpu = Config.gpu

class Inference:
    def __init__(self, seq):
        self.seq=seq
        self.N=len(self.seq)
        # FM_inside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        # FM_outside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]

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

    # def ComputeInside_Parallel(self, i, j, inside1, inside2, inside3):
    def ComputeInside_Parallel(self, i, j):
        x = F.concat((FM_inside[i][j-1] , FM_inside[i+1][j-1] , FM_inside[i+1][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
        # x = F.concat((inside1, inside2, inside3, self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
        return self.model(x)

    def ComputeOutside_Parallel(self, i, j, outside1, outside2, outside3):
        # a = i-1
        # b = j+1
        # if i == 0:
        #     a = self.N-1
        # if j == self.N-1:
        #     b = 0
        x = F.concat((outside1, outside2, outside3, self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
        return self.model(x)

    def wrapper_inside(self,args):
        return self.ComputeInside_Parallel(*args)

    def wrapper_outside(self,args):
        return self.ComputeOutside_Parallel(*args)

    def Compute_Inside(self):
        #define feature matrix
        FM_inside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        #compute inside
        for n in range(1,self.N):
            # start = time()
            #Parallel
            if PARALLEL == True and self.N-n > 300:
                #multiprocessing
                # process = math.floor((self.N-n)/140)
                # p = Pool(process)
                # # r = p.map(self.wrapper_inside, [[j-n, j, FM_inside[j-n][j-1] , FM_inside[j-n+1][j-1] , FM_inside[j-n+1][j]] for j in range(n,self.N)])
                # r = p.map(self.wrapper_inside, [[j-n, j] for j in range(n, self.N)])
                # p.close()
                # for j in range(n,self.N):
                #     FM_inside[j-n][j] = r.pop(0)

                #joblib
                # r = Parallel(n_jobs=16)( [delayed(self.ComputeInside_Parallel)(j-n, j, FM_inside[j-n][j-1] , FM_inside[j-n+1][j-1] , FM_inside[j-n+1][j]) for j in range(n,self.N)])
                # for j in range(n,self.N):
                #     FM_inside[j-n][j] = r.pop(0)

                #multithread
                # pool = ThreadPoolExecutor(4)
                # r = [pool.submit(self.ComputeInside_Parallel, j-n, j, FM_inside[j-n][j-1] , FM_inside[j-n+1][j-1] , FM_inside[j-n+1][j]) for j in range(n,self.N)]
                # for j in range(n,self.N):
                #     FM_inside[j-n][j] = r[j].result()
                # pool.shutdown()
                print("PARALLEL inside")



            else:
                for j in range(n,self.N):
                    i = j-n
                    x = F.concat((FM_inside[i][j-1] , FM_inside[i+1][j-1] , FM_inside[i+1][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                    FM_inside[i][j] = self.model(x)

        return FM_inside

    def Compute_Outside(self):
        FM_outside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        # compute outside
        for n in range(self.N-1 , 1-1 , -1):
            if PARALLEL == True and self.N-n > 50:
                # multiprocessing
                # if self.N-n > multi.cpu_count():
                #     p = Pool(multi.cpu_count())
                # else:
                #     p = Pool(self.N-n)
                # p = Pool(round((self.N-n)/10))
                # p = Pool(4)
                # for j in range(self.N-1 , n-1 , -1):
                #     i = j-n
                #     a = i-1
                #     b = j+1
                #     if i == 0:
                #         a = self.N-1
                #     if j == self.N-1:
                #         b = 0
                # r = p.map(self.wrapper_outside, [[j-n, j, FM_outside[i][b] , FM_outside[a][b] , FM_outside[a][j] for j in range(self.N-1 , n-1 , -1)])
                # p.close()
                # for j in range(self.N-1 , n-1 , -1):
                #     FM_outside[j-n][j] = r.pop(0)
                #
                # joblib
                # r = Parallel(n_jobs=16)( [delayed(self.ComputeOutside_Parallel)(j-n, j, FM_outside[i][b] , FM_outside[a][b] , FM_outside[a][j]) for j in range(self.N-1 , n-1 , -1)])
                # for j in range(self.N-1 , n-1 , -1):
                #     FM_outside[j-n][j] = r.pop(0)
                #
                # multithread
                # pool = ThreadPoolExecutor(4)
                # r = [pool.submit(self.ComputeOutside_Parallel, j-n, j, FM_outside[i][b] , FM_outside[a][b] , FM_outside[a][j]) for j in range(self.N-1 , n-1 , -1)]
                # for j in range(self.N-1 , n-1 , -1):
                #     FM_outside[j-n][j] = r[j].result()
                # pool.shutdown()
                print("PARALLEL outside")

            else:
                for j in range(self.N-1 , n-1 , -1):
                    i = j-n
                    a = i-1
                    b = j+1
                    if i == 0:
                        a = self.N-1
                    if j == self.N-1:
                        b = 0
                    x = F.concat((FM_outside[i][b] , FM_outside[a][b] , FM_outside[a][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                    FM_outside[i][j] = self.model(x)
        return FM_outside

    def wrapper_insideoutside(self, changer):
        if changer == "inside":
            return self.Compute_Inside()
        elif changer == "outside":
            return self.Compute_Outside()
        else:
            print("error")
            return

    def ComputeInsideOutside(self, model):
        self.model = model
        BP = [[0 for i in range(self.N)] for j in range(self.N)]
        FM_inside = self.Compute_Inside()
        FM_outside = self.Compute_Outside()

        # p = Pool(2)
        # FM_inside, FM_outside = p.map(self.wrapper_insideoutside, [j for j in ["inside", "outside"]])
        # p.close()

        # gpu_device = 0
        # cuda.get_device(gpu_device).use()
        # model.to_gpu(gpu_device)
        # xp = cuda.cupy

        #define feature matrix
        #use gpu
        # if gpu == True:
        #     print("gpu")
        #     # FM_inside = [[Variable(xp.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        #     # FM_outside = [[Variable(xp.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]


        # marge inside outside
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                x = F.concat((FM_inside[i][j] , FM_outside[i][j]) ,axis=1)
                BP[i][j] = self.model(x , inner = False)
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
