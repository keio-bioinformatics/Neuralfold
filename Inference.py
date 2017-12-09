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
base_length = Config.base_length

class Inference:
    def __init__(self, seq):
        self.seq=seq
        self.N=len(self.seq)
        # FM_inside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        # FM_outside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        self.sequence_vector = np.empty((0,base_length), dtype=np.float32)
        for base in self.seq:
            self.sequence_vector = np.vstack((self.sequence_vector, self.base_represent(base)))
        self.sequence_vector = Variable(self.sequence_vector)

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
                return np.array([[1,0,0,0]] , dtype=np.float32)
            elif base in ['U' ,'u']:
                return np.array([[0,1,0,0]] , dtype=np.float32)
            elif base in ['G' ,'g']:
                return np.array([[0,0,1,0]] , dtype=np.float32)
            elif base in ['C' ,'c']:
                return np.array([[0,0,0,1]] , dtype=np.float32)
            else:
                # print(base)
                return np.array([[0,0,0,0]] , dtype=np.float32)

        # else:
        #     if base in ['A' ,'a']:
        #         return Variable(np.array([[1,0,0,0]] , dtype=np.float32))
        #     elif base in ['U' ,'u']:
        #         return Variable(np.array([[0,1,0,0]] , dtype=np.float32))
        #     elif base in ['G' ,'g']:
        #         return Variable(np.array([[0,0,1,0]] , dtype=np.float32))
        #     elif base in ['C' ,'c']:
        #         return Variable(np.array([[0,0,0,1]] , dtype=np.float32))
        #     else:
        #         # print(base)
        #         return Variable(np.array([[0,0,0,0]] , dtype=np.float32))

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

    def hash_for_inside(self, i, j):
        n = j-i
        return int(n * (self.N + self.N - n +1)/2 + i)

    def hash_for_outside(self, i, j):
        n = j-i
        index_full = (self.N*(self.N+1)/2) - 1
        return int(index_full - (n * (self.N + self.N - n +1)/2 + i))

    def hash_for_BP(self, i, j):
        n = j-i
        return int(n * (self.N + self.N - n +1)/2 + i)

    def Compute_Inside(self):
        #define feature matrix
        # FM_inside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        FM_inside = Variable(np.zeros((self.N, 1,FEATURE_SIZE), dtype=np.float32))

        #compute inside
        for n in range(1,self.N):
            start = time()
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
                #
                # for j in range(n,self.N):
                #     i = j-n
                #     x = F.concat((FM_inside[i][j-1] , FM_inside[i+1][j-1] , FM_inside[i+1][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                #     FM_inside[i][j] = self.model(x)
                #

                # batch
                # start = time()
                # x = Variable(np.array([[]], dtype=np.float32))
                # for j in range(n,self.N):
                #     i = j-n
                #     x = F.concat((x, FM_inside[i][j-1] , FM_inside[i+1][j-1] , FM_inside[i+1][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                # x = x.reshape(self.N-n, -1)
                # r = self.model(x).reshape(self.N-n,1,FEATURE_SIZE)
                # for j in range(n,self.N):
                #     FM_inside[j-n][j] = r[j-n]

                # start = time()
                # x = Variable(np.array([[]], dtype=np.float32))
                # if n == 1:
                #     for j in range(n,self.N):
                #         i = j-n
                #         x = F.concat((x, FM_inside[self.hash_for_inside(i,j-1)] , Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)), FM_inside[self.hash_for_inside(i+1,j)] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                # else:
                #     for j in range(n,self.N):
                #         i = j-n
                #         x = F.concat((x, FM_inside[self.hash_for_inside(i,j-1)] , FM_inside[self.hash_for_inside(i+1,j-1)] , FM_inside[self.hash_for_inside(i+1,j)] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                # x = x.reshape(self.N-n, -1)
                # FM_inside = F.vstack((FM_inside, self.model(x).reshape(self.N-n,1,FEATURE_SIZE)))

                # start_inside_row = time()
                if n == 1:
                    x = F.hstack((FM_inside[self.hash_for_inside(0, n-1) : self.hash_for_inside(self.N-n, self.N-1)], Variable(np.zeros((self.N-1, 1,FEATURE_SIZE), dtype=np.float32)),
                        FM_inside[self.hash_for_inside(1, n) : self.hash_for_inside(0, n)])).reshape(self.N-n, -1)
                else:
                    x = F.hstack((FM_inside[self.hash_for_inside(0, n-1) : self.hash_for_inside(self.N-n, self.N-1)], FM_inside[self.hash_for_inside(1, n-1) : self.hash_for_inside(self.N-n+1, self.N-1)],
                        FM_inside[self.hash_for_inside(1, n) : self.hash_for_inside(0, n)])).reshape(self.N-n, -1)
                x = F.hstack((x,self.sequence_vector[0 : self.N-n], self.sequence_vector[n : self.N]))
                FM_inside = F.vstack((FM_inside, self.model(x).reshape(self.N-n,1,FEATURE_SIZE)))
                # print(' inside_row : '+str(time() - start_inside_row)+'sec')
        return FM_inside



    def Compute_Outside(self):
        # define feature matrix
        # FM_outside = [[Variable(np.zeros((1,FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        FM_outside = Variable(np.empty((0, 1, FEATURE_SIZE), dtype=np.float32))

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
                # for j in range(self.N-1 , n-1 , -1):
                #     i = j-n
                #     a = i-1
                #     b = j+1
                #     if i == 0:
                #         a = self.N-1
                #     if j == self.N-1:
                #         b = 0
                #     x = F.concat((FM_outside[i][b] , FM_outside[a][b] , FM_outside[a][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                #     FM_outside[i][j] = self.model(x)
                #
                #batch
                # x = Variable(np.array([[]], dtype=np.float32))
                # for j in range(self.N-1 , n-1 , -1):
                #     i = j-n
                #     a = i-1
                #     b = j+1
                #     if i == 0:
                #         a = self.N-1
                #     if j == self.N-1:
                #         b = 0
                #     x = F.concat((x, FM_outside[i][b] , FM_outside[a][b] , FM_outside[a][j] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                # x = x.reshape((self.N-1)-(n-1), -1)
                # r = self.model(x).reshape((self.N-1)-(n-1), 1, FEATURE_SIZE)
                # for j in range(self.N-1 , n-1 , -1):
                #     FM_outside[j-n][j] = r[j-n]


                # start_outside_row = time()
                if n == self.N-1:
                    x = F.hstack((Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32)), Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32)),
                        Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32)))).reshape(self.N-n, -1)
                elif n == self.N-2:
                    first = F.vstack((Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32)), FM_outside[0].reshape(1,1,FEATURE_SIZE)))
                    second = F.vstack((Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32)), Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32))))
                    third = F.vstack((FM_outside[0].reshape(1,1,FEATURE_SIZE), Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32))))
                    x = F.hstack((first, second, third)).reshape(self.N-n, -1)
                else:
                    first = F.vstack((Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32)), FM_outside[self.hash_for_outside(self.N-n-2, self.N-1) : self.hash_for_outside(self.N-n-1, self.N-1)]))
                    second = F.vstack((Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32)), FM_outside[self.hash_for_outside(self.N-n-3, self.N-1) : self.hash_for_outside(self.N-n-2, self.N-1)], Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32))))
                    third = F.vstack((FM_outside[self.hash_for_outside(self.N-n-2, self.N-1) : self.hash_for_outside(self.N-n-1, self.N-1)], Variable(np.zeros((1, 1,FEATURE_SIZE), dtype=np.float32))))
                    x = F.hstack((first, second, third)).reshape(self.N-n, -1)
                x = F.hstack((x,self.sequence_vector[self.N-n-1 :: -1], self.sequence_vector[self.N-1 : n-1 : -1]))
                FM_outside = F.vstack((FM_outside, self.model(x).reshape(self.N-n,1,FEATURE_SIZE)))
                # print(' outside_row : '+str(time() - start_outside_row)+'sec')

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
        # BP = [[0 for i in range(self.N)] for j in range(self.N)]
        BP = Variable(np.zeros((self.N, 1,1), dtype=np.float32))
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
            #batch
            # x = Variable(np.array([[]], dtype=np.float32))
            # for j in range(n,self.N):
            #     i = j-n
            #     x = F.concat((x, FM_inside[i][j] , FM_outside[i][j]) ,axis=1)
            # x = x.reshape(self.N-n, -1)
            # r = self.model(x , inner = False).reshape(self.N-n,1,1)
            # for j in range(n,self.N):
            #     BP[j-n][j] = r[j-n]
            for j in range(n,self.N):
                i = j-n
                if n == self.N-1:
                    x = F.hstack((FM_inside[self.hash_for_inside(0, n) : ], FM_outside[self.hash_for_outside(0, n) : : -1]))
                else:
                    x = F.hstack((FM_inside[self.hash_for_inside(0, n) : self.hash_for_inside(0, n+1)], FM_outside[self.hash_for_outside(0, n) : self.hash_for_outside(0, n+1) : -1]))
                BP = F.vstack((BP, self.model(x, inner  = False).reshape(self.N-n,1,1)))
        # print(BP[3].data+1)
        return BP

    def buildDP(self, BP):
        DP = np.zeros((self.N,self.N))
        for n in range(1,self.N):
            for j in range(n,self.N):
                i = j-n
                if self.pair_check((self.seq[i].upper(), self.seq[j].upper())):
                    case1 = DP[i+1,j-1] + BP[self.hash_for_BP(i, j)].data
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
            elif DP[i, j] == DP[i+1, j-1]+BP[self.hash_for_BP(i, j)].data:
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
