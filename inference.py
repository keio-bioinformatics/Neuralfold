import numpy as np
import Config
min_loop_length = Config.min_loop_length
class Inference:
    def __init__(self,seq):
        self.seq=seq
        self.N=len(self.seq)

    #initialize matrix with zeros where can't have pairings
    def initializeBP(self):
        #NxN matrix that stores the BaseParing probability
        #     BP = np.empty((self.N,self.N))
        #     BP[:] = np.NAN
        BP = np.zeros((self.N,self.N))
        return BP


    #initialize Feature matrix
    def initializeFM(self):

    def ComputeInsideOutside(self):
        BP=initializeBP()
        #FM=initializeFM()
        return BP

    # def ComputeBP(self):

    def ComputePosterior(self,BP):
        DP = np.zeros((self.N,self.N))
        for n in xrange(1,self.N):
            for j in xrange(n,self.N):
                i = j-n;
                case1 = DP[i+1,j-1]+BP(seq[i],seq[j]);
                case2 = DP[i+1,j]
                case3 = DP[i,j-1]
                if i+3<=j:
                    tmp=[];
                    for k in xrange(i+1,j):
                        tmp.append(s[i,k]+s[k+1,j]);
                    case4=max(tmp);
                    DP[i,j]=max(case1,case2,case3,case4);
                else:
                    DP[i,j]=max(case1,case2,case3);
        return DP
