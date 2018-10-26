import numpy as np
from . import Config
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
from time import time
import math
import random
import pulp

min_loop_length = Config.min_loop_length
# self.FEATURE_SIZE = Config.feature_length
bifurcation = Config.bifurcation
PARALLEL = Config.Parallel
gpu = Config.gpu
base_length = Config.base_length

class Inference:
    def __init__(self, seq, FEATURE_SIZE=80):
        self.seq=seq
        self.N=len(self.seq)

        self.sequence_vector = np.empty((0,base_length), dtype=np.float32)
        for base in self.seq:
            self.sequence_vector = np.vstack((self.sequence_vector, self.base_represent(base)))
        self.sequence_vector = Variable(self.sequence_vector)
        self.FEATURE_SIZE = FEATURE_SIZE

    # def pair_check(self,tup):
    #     if tup in [('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'),('G','U'),('U','G')]:
    #         return True
    #     return False

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
                return np.array([[1,0,0,0,0]] , dtype=np.float32)
            elif base in ['U' ,'u']:
                return np.array([[0,1,0,0,0]] , dtype=np.float32)
            elif base in ['G' ,'g']:
                return np.array([[0,0,1,0,0]] , dtype=np.float32)
            elif base in ['C' ,'c']:
                return np.array([[0,0,0,1,0]] , dtype=np.float32)
            else:
                # print(base)
                return np.array([[0,0,0,0,0]] , dtype=np.float32)

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
        # FM_inside = [[Variable(np.zeros((1,self.FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        FM_inside = Variable(np.zeros((self.N, 1,self.FEATURE_SIZE), dtype=np.float32))
        start_inside = time()

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
                # r = self.model(x).reshape(self.N-n,1,self.FEATURE_SIZE)
                # for j in range(n,self.N):
                #     FM_inside[j-n][j] = r[j-n]

                # start = time()
                # x = Variable(np.array([[]], dtype=np.float32))
                # if n == 1:
                #     for j in range(n,self.N):
                #         i = j-n
                #         x = F.concat((x, FM_inside[self.hash_for_inside(i,j-1)] , Variable(np.zeros((1,self.FEATURE_SIZE), dtype=np.float32)), FM_inside[self.hash_for_inside(i+1,j)] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                # else:
                #     for j in range(n,self.N):
                #         i = j-n
                #         x = F.concat((x, FM_inside[self.hash_for_inside(i,j-1)] , FM_inside[self.hash_for_inside(i+1,j-1)] , FM_inside[self.hash_for_inside(i+1,j)] , self.base_represent(self.seq[i]) , self.base_represent(self.seq[j])) ,axis=1)
                # x = x.reshape(self.N-n, -1)
                # FM_inside = F.vstack((FM_inside, self.model(x).reshape(self.N-n,1,self.FEATURE_SIZE)))

                # start_inside_row = time()
                if n == 1:
                    x = F.hstack((FM_inside[self.hash_for_inside(0, n-1) : self.hash_for_inside(self.N-n, self.N-1)], Variable(np.zeros((self.N-1, 1,self.FEATURE_SIZE), dtype=np.float32)),
                        FM_inside[self.hash_for_inside(1, n) : self.hash_for_inside(0, n)])).reshape(self.N-n, -1)
                else:
                    x = F.hstack((FM_inside[self.hash_for_inside(0, n-1) : self.hash_for_inside(self.N-n, self.N-1)], FM_inside[self.hash_for_inside(1, n-1) : self.hash_for_inside(self.N-n+1, self.N-1)],
                        FM_inside[self.hash_for_inside(1, n) : self.hash_for_inside(0, n)])).reshape(self.N-n, -1)
                x = F.hstack((x,self.sequence_vector[0 : self.N-n], self.sequence_vector[n : self.N]))
                # if n % random.randint(20,25) == 0:
                # if random.randint(0,25) == 5:
                #     x.unchain_backward()
                FM_inside = F.vstack((FM_inside, self.model(x).reshape(self.N-n,1,self.FEATURE_SIZE)))
                # print(' inside_row : '+str(time() - start_inside_row)+'sec')
        # print(' inside : '+str(time() - start_inside)+'sec')
        return FM_inside



    def Compute_Outside(self):
        # define feature matrix
        # FM_outside = [[Variable(np.zeros((1,self.FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        FM_outside = Variable(np.empty((0, 1, self.FEATURE_SIZE), dtype=np.float32))
        start_outside = time()

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
                # r = self.model(x).reshape((self.N-1)-(n-1), 1, self.FEATURE_SIZE)
                # for j in range(self.N-1 , n-1 , -1):
                #     FM_outside[j-n][j] = r[j-n]


                # start_outside_row = time()
                if n == self.N-1:
                    x = F.hstack((Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)), Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)),
                        Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)))).reshape(self.N-n, -1)
                elif n == self.N-2:
                    first = F.vstack((Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)), FM_outside[0].reshape(1,1,self.FEATURE_SIZE)))
                    second = F.vstack((Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)), Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32))))
                    third = F.vstack((FM_outside[0].reshape(1,1,self.FEATURE_SIZE), Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32))))
                    x = F.hstack((first, second, third)).reshape(self.N-n, -1)
                else:
                    first = F.vstack((Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)), FM_outside[self.hash_for_outside(self.N-n-2, self.N-1) : self.hash_for_outside(self.N-n-1, self.N-1)]))
                    second = F.vstack((Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)), FM_outside[self.hash_for_outside(self.N-n-3, self.N-1) : self.hash_for_outside(self.N-n-2, self.N-1)], Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32))))
                    third = F.vstack((FM_outside[self.hash_for_outside(self.N-n-2, self.N-1) : self.hash_for_outside(self.N-n-1, self.N-1)], Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32))))
                    x = F.hstack((first, second, third)).reshape(self.N-n, -1)
                x = F.hstack((x,self.sequence_vector[self.N-n-1 :: -1], self.sequence_vector[self.N-1 : n-1 : -1]))
                # if n % random.randint(20,25) == 0:
                # if random.randint(0,25) == 5:
                #     x.unchain_backward()
                FM_outside = F.vstack((FM_outside, self.model(x).reshape(self.N-n,1,self.FEATURE_SIZE)))
                # print(' outside_row : '+str(time() - start_outside_row)+'sec')
        # print(' outside : '+str(time() - start_outside)+'sec')
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
        BP = Variable(np.zeros((self.N, 1,1), dtype=np.float32))
        Variable(np.zeros((1, 1,self.FEATURE_SIZE), dtype=np.float32)),
        FM_inside = self.Compute_Inside()
        FM_outside = self.Compute_Outside()

        # p = Pool(2)
        # FM_inside, FM_outside = p.map(self.wrapper_insideoutside, [j for j in ["inside", "outside"]])
        # p.close()

        # gpu_device = 0
        # cuda.get_device(gpu_device).use()
        # model.to_gpu(gpu_device)
        # xp = cuda.cupy
        #
        #define feature matrix
        #use gpu
        # if gpu == True:
        #     print("gpu")
        #     # FM_inside = [[Variable(xp.zeros((1,self.FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]
        #     # FM_outside = [[Variable(xp.zeros((1,self.FEATURE_SIZE), dtype=np.float32)) for i in range(self.N)] for j in range(self.N)]


        # merge inside outside

        start_merge = time()
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
            if n == self.N-1:
                x = F.hstack((FM_inside[self.hash_for_inside(0, n) : ], FM_outside[self.hash_for_outside(0, n) : : -1])).reshape(self.N-n, -1)
            else:
                x = F.hstack((FM_inside[self.hash_for_inside(0, n) : self.hash_for_inside(0, n+1)], FM_outside[self.hash_for_outside(0, n) : self.hash_for_outside(0, n+1) : -1])).reshape(self.N-n, -1)
            # BP = F.vstack((BP, self.model(x, inner  = False).reshape(self.N-n,1,1)))
            BP = F.vstack((BP, self.model(x, inner  = False).reshape(self.N-n,1,1)))

        # print(' merge : '+str(time() - start_merge)+'sec')
        return BP


    def ComputeNeighbor(self, model):
        # set model
        self.model = model
        neighbor = model.neighbor

        # initiate BPP table
        BP = Variable(np.zeros((self.N, 1,1), dtype=np.float32))

        # Fill the side with 0 vectors
        sequence_vector_neighbor = self.sequence_vector.reshape(self.N,base_length)
        sequence_vector_neighbor = F.vstack((sequence_vector_neighbor, Variable(np.zeros((neighbor, base_length), dtype=np.float32)) ))
        sequence_vector_neighbor = F.vstack((Variable(np.zeros((neighbor, base_length), dtype=np.float32)) , sequence_vector_neighbor))

        # Create vectors around each base
        side_input_size =  base_length * (neighbor * 2 +1)
        side_input_vector = Variable(np.empty((0, side_input_size), dtype=np.float32))
        for base in range(0, self.N):
            side_input_vector = F.vstack((side_input_vector, sequence_vector_neighbor[base:base+(neighbor * 2 + 1)].reshape(1,side_input_size)))
        # all_input_vector = Variable(np.empty((0, side_input_size*2), dtype=np.float32))

        # predict BPP
        for interval in range(1, self.N):
            left = side_input_vector[0 : self.N - interval]
            right = side_input_vector[interval : self.N]
            x = F.hstack((left, right))

            # If there is an overlap, fill that portion with a specific vector
            if interval <= neighbor:
                row = self.N-interval
                column = (neighbor-interval+1)*2*base_length
                a = int(column/base_length)
                surplus = Variable(np.array(([0,0,0,0,1] * row * a), dtype=np.float32))
                surplus = surplus.reshape(row, column)
                x[0:row,side_input_size-int(column/2)+1:side_input_size+int(column/2)+1].data = surplus.data

            # add location information
            direct = False
            onehot = True
            if direct:
                left_location = Variable(np.arange(1, self.N - interval+1, dtype=np.float32).reshape(self.N - interval,1))/10
                right_location = Variable(np.arange(interval+1, self.N+1, dtype=np.float32).reshape(self.N - interval,1))/10
                full_length = Variable(np.full(self.N - interval, self.N, dtype=np.float32).reshape(self.N - interval,1))/10
                x = F.hstack((x, left_location, right_location, full_length))
            elif onehot:
                # left_location = Variable(np.eye(80)[np.ceil(np.arange(1, self.N - interval+1, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                # right_location = Variable(np.eye(80)[np.ceil(np.arange(interval+1, self.N+1, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                # full_length = Variable(np.eye(80)[np.ceil(np.full(self.N - interval, self.N, dtype=np.float32)/10).astype(np.int8)].astype(np.float32))
                interval_length = Variable(np.eye(20)[np.ceil(np.full(self.N - interval, math.log(interval,1.5), dtype=np.float32)).astype(np.int8)].astype(np.float32))
                # x = F.hstack((x, left_location, right_location, full_length))
                x = F.hstack((x, interval_length))

            BP = F.vstack((BP, self.model(x).reshape(self.N - interval,1,1)))
        return BP

    # def buildDP2(self, BP):
    #     DP = np.zeros((self.N,self.N))
    #     for n in range(1,self.N):
    #         for j in range(n,self.N):
    #             i = j-n
    #             if self.pair_check((self.seq[i].upper(), self.seq[j].upper())):
    #                 case1 = DP[i+1,j-1] + BP[self.hash_for_BP(i, j)].data
    #             else:
    #                 case1 = -1000
    #             case2 = DP[i+1,j]
    #             case3 = DP[i,j-1]
    #             if i+3<=j:
    #                 tmp = np.array([])
    #                 for k in range(i+1,j):
    #                     tmp = np.append(tmp , DP[i,k] + DP[k+1,j])
    #                 case4 = np.max(tmp)
    #                 DP[i,j] = max(case1,case2,case3,case4)
    #             else:
    #                 DP[i,j] = max(case1,case2,case3)
    #     return DP

    def canonical_basepairs(self):
        sequence = np.array(list(self.seq.upper()))
        bases= ['A', 'U', 'G', 'C']
        matrix = []
        matrix_reverse = []
        for base in bases:
            a = np.zeros((self.N), dtype=np.int8)
            a[sequence == base] = 1
            a[sequence != base] = 0
            a = np.tile(a, (self.N, 1))
            matrix.append(a)
            matrix_reverse.append(a.transpose())
        all_bases = np.zeros((self.N,self.N), dtype=np.int8)
        # if tup in [('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'),('G','U'),('U','G')]:
        for tup in [(0, 1), (1, 0), (2, 3), (3, 2),(1,2),(2,1)]:
            all_bases += matrix[tup[0]] * matrix_reverse[tup[1]]
        return all_bases


    def buildDP(self, BP, gamma, margin=None):
        DP = np.zeros((self.N,self.N), dtype=np.float32)
        TP = np.empty((self.N,self.N), dtype=np.int)

        DP_bifarcation_row = np.zeros((self.N,self.N), dtype=np.float32)
        DP_bifarcation_column = np.zeros((self.N,self.N), dtype=np.float32)


        canonical_base_matrix = self.canonical_basepairs()

        for n in range(1,self.N):
            p = BP[self.hash_for_BP(0, n):self.hash_for_BP(0, n+1)].reshape(self.N-n)

            s = canonical_base_matrix[range(0,self.N-n),range(n,self.N)] * (gamma + 1) * p - 1
            if margin is not None:
                s += margin[range(0,self.N-n),range(n,self.N)]

            case1 = DP[range(1,self.N-n+1),range(n-1,self.N-1)] + s
            case2 = DP[range(1,self.N-n+1),range(n,self.N)]
            case3 = DP[range(0,self.N-n),range(n-1,self.N-1)]
            # if n <= 3:
            #     case1=case1-100
            if n >= 3:
                case4 = DP_bifarcation_row[1:n, 0:self.N-n] + DP_bifarcation_column[self.N-n+1:self.N, n:self.N]
                DP_diagonal = np.vstack((case1, case2, case3, case4)).max(axis=0)
                TP_diagonal = np.vstack((case1, case2, case3, case4)).argmax(axis=0)
            else:
                DP_diagonal = np.vstack((case1, case2, case3)).max(axis=0)
                TP_diagonal = np.vstack((case1, case2, case3)).argmax(axis=0)

            DP[range(0,self.N-n),range(n,self.N)] = DP_diagonal
            TP[range(0,self.N-n),range(n,self.N)] = TP_diagonal

            DP_bifarcation_row[n,0:self.N-n] = DP_diagonal
            DP_bifarcation_column[self.N-n-1,n:self.N] = DP_diagonal

        return DP,TP


    # def traceback2(self, DP, BP, i, j, pair):
    #     if i < j:
    #         # if DP[i, j] == DP[i+1, j]:
    #         if np.absolute(DP[i, j] - DP[i+1, j]) < 0.0001:
    #             pair = self.traceback2(DP, BP, i+1, j, pair)
    #         # elif DP[i, j] == DP[i, j-1]:
    #         elif np.absolute(DP[i, j] - DP[i, j-1]) < 0.0001:
    #             pair = self.traceback2(DP, BP, i, j-1, pair)
    #         # elif DP[i, j] == DP[i+1, j-1]+BP[self.hash_for_BP(i, j)].data:
    #         # elif np.absolute(DP[i, j] - (DP[i+1, j-1]+BP[self.hash_for_BP(i, j)].data)) < 0.0001:
    #         elif np.absolute(DP[i, j] - (DP[i+1, j-1]+BP[self.hash_for_BP(i, j),0,1].data)) < 0.0001:
    #             pair = np.append(pair, [[i,j]], axis=0)
    #             pair = self.traceback2(DP, BP, i+1, j-1, pair)
    #         else:
    #             for k in range(i+1,j):
    #                 # if DP[i,j] == DP[i,k]+DP[k+1,j]:
    #                 if np.absolute(DP[i,j] - (DP[i,k]+DP[k+1,j])) < 0.0001:
    #                     pair = self.traceback2(DP, BP, i, k, pair)
    #                     pair = self.traceback2(DP, BP, k+1, j, pair)
    #                     break
    #             print("can not traceback")
    #             # print(np.absolute(DP[i, j] - DP[i+1, j]))
    #             # print(np.absolute(DP[i, j] - DP[i, j-1]))
    #
    #     return pair


    def traceback(self, TP, i, j, pair):
        if i < j:
            if TP[i,j] == 0:
                pair.append((i,j))
                pair = self.traceback(TP, i+1, j-1, pair)
            elif TP[i,j] == 1:
                pair = self.traceback(TP, i+1, j, pair)
            elif TP[i,j] == 2:
                pair = self.traceback(TP, i, j-1, pair)
            else:
                pair = self.traceback(TP, i, TP[i,j]-3+i+1, pair)
                pair = self.traceback(TP, TP[i,j]-3+i+1+1, j, pair)
        return pair


    def nussinov(self, BP, gamma, margin=None):
        DP, TP = self.buildDP(BP, gamma=gamma, margin=margin)
        pair = self.traceback(TP, 0, self.N-1, [])
        return pair


    def ipknot(self, bp, gamma,
               margin=None, disable_th=False, allowed_bp=None, non_canonical=False):
        prob = pulp.LpProblem("IPknot", pulp.LpMaximize)
        seqlen = self.N
        nlevels = len(gamma)

        if disable_th or max(gamma)+1 >= seqlen or max(gamma) <= 0:
            enabled_bp = np.ones((seqlen, seqlen), dtype=np.int8) * 2

        else: # acceleration by 'at most gamma+1' candidates for each base
            # reshape BP matrix
            bp_reshaped = np.zeros((seqlen, seqlen))
            j = 0
            for i in range(seqlen, 0, -1):
                bp_reshaped += np.diag(bp[j:j+i].reshape((i)), k=seqlen-i)
                if seqlen-i>0:
                    bp_reshaped += np.diag(bp[j:j+i].reshape((i)), k=-(seqlen-i))
                j += i

            # calculate ranks
            sorted_bp_idx = np.argsort(bp_reshaped,axis=1)[:, ::-1]
            enabled_bp = np.zeros((seqlen, seqlen), dtype=np.int8)
            for i in range(seqlen):
                for j in range(int(max(gamma)+1)): # approximate threshold cut
                    enabled_bp[i, sorted_bp_idx[i, j]] = 1
            enabled_bp += np.transpose(enabled_bp)

        if allowed_bp is not None:
            a = np.zeros((seqlen, seqlen), dtype=np.int8)
            for p in allowed_bp:
                a[p[0], p[1]] = a[p[1], p[0]] = 1
            enabled_bp *= a

        if not non_canonical:
            enabled_bp *= self.canonical_basepairs()

        # variables and objective function
        x = [[[None for i in range(seqlen)] for j in range(seqlen)] for k in range(nlevels)]
        obj = []
        th = 0 if disable_th else 1
        for k in range(nlevels):
            for j in range(seqlen):
                for i in range(j):
                    s = (gamma[k]+1) * bp[self.hash_for_BP(i, j)] - th
                    if margin is not None:
                        s += margin[i,j]
                    if enabled_bp[i, j] >= 2:
                        x[k][i][j] = x[k][j][i] = pulp.LpVariable('x[%d][%d][%d]' % (k, i, j), 0, 1, 'Binary')
                        if type(s) is  Variable:
                            s = s.data
                        obj.append(s * x[k][i][j])
        prob += pulp.lpSum(obj)

        # constraints 1
        for i in range(seqlen):
            prob += pulp.lpSum([x[k][i][j] for j in range(seqlen) for k in range(nlevels) if x[k][i][j] is not None]) <= 1

        # constraints 2
        for k in range(nlevels):
            for j2 in range(seqlen):
                for j1 in range(j2):
                    for i2 in range(j1):
                        if x[k][i2][j2] is not None:
                            for i1 in range(i2):
                                if x[k][i1][j1] is not None:
                                    prob += pulp.lpSum(x[k][i1][j1] + x[k][i2][j2]) <= 1

        # constraints 3
        for k2 in range(nlevels):
            for j2 in range(seqlen):
                for i2 in range(j2):
                    if x[k2][i2][j2] is not None:
                        for k1 in range(k2):
                            # o = 0
                            c1 = [x[k1][i1][j1] for i1 in range(i2+1, j2) for j1 in range(i2-1) if x[k1][j1][j1] is not None]
                            c2 = [x[k1][i1][j1] for i1 in range(i2+1, j2) for j1 in range(j2+1, seqlen) if x[k1][i1][j1] is not None]
                            prob += pulp.lpSum(c1+c2) >= x[k2][i2][j2]

        # solve the IP problem
        start_ip = time()
        try:
            prob.solve()
            # prob.solve(CPLEX(msg=False))
        except KeyError as e:
            return []

        pair = [ [] for _ in range(nlevels) ]
        for k in range(nlevels):
            for j in range(seqlen):
                for i in range(j):
                    if x[k][i][j] is not None and x[k][i][j].value() == 1:
                        pair[k].append((i,j))
        return pair


    def dot_parentheis(self, pair):
        if len(pair) == 0:
            return ''

        elif len(pair[0]) > 0 and type(pair[0][0]) is int: # nussinov
            y = ['.']*len(self.seq)
            for p in pair:
                y[p[0]] = '('
                y[p[1]] = ')'
            return "".join(y)

        else: # ipknot
            l = ('(', '[', '{', '<')
            r = (')', ']', '}', '>')
            y = ['.']*len(self.seq)
            for k, kpair in enumerate(pair):
                for p in kpair:
                    y[p[0]] = l[k]
                    y[p[1]] = r[k]
            return "".join(y)


    def calculate_score(self,  BP, ipknot, pair, prediction, gamma, margin=None):
        s = np.zeros((1,1), dtype=np.float32)
        if type(BP) is Variable:
            s = Variable(s)

        if len(pair) == 0:
            pass

        elif ipknot: # ipknot
            # print(pair)
            if prediction==False:
                pair = self.ipknot(BP, gamma, disable_th=True, allowed_bp=pair)
            # print(pair)
            for k, kpair in enumerate(pair):
                for p in kpair:
                    s += (gamma[k]+1) * BP[self.hash_for_BP(p[0], p[1])] - 1.
                    if margin is not None:
                        s += margin[p[0], p[1]]


        # elif len(pair[0]) > 0 and type(pair[0][0]) is int: # nussinov
        # elif type(gamma) is not  list: # nussinov
        else: # nussinov
            for p in pair:
                s += (gamma+1.) * BP[self.hash_for_BP(p[0], p[1])] - 1.
                if margin is not None:
                    s += margin[p[0], p[1]]


        return s.reshape(1,)


    def ComputePosterior(self, BP, ipknot, gamma, margin=None):
        if ipknot:
            pair = self.ipknot(BP, gamma, margin)
        else:
            pair = self.nussinov(BP, gamma, margin)

        return pair