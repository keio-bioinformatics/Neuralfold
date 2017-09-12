import numpy as np
import Config
min_loop_length = Config.min_loop_length
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

    def ComputeInsideOutside(self):
        BP = self.initializeBP()
        #FM=initializeFM()
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
