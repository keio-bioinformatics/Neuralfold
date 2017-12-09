import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
from tqdm import tqdm
from time import time
import Test
import Evaluate
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
import random
import os
from multiprocessing import Pool
import multiprocessing as multi
from operator import itemgetter
import math
from joblib import Parallel, delayed

iters_num = Config.iters_num
TEST = Config.TEST
batch_num = Config.batch_num
maximum_slots = Config.maximum_slots
BATCH = Config.BATCH
Iterative_Parameter_Mixture = Config.Iterative_Parameter_Mixture

class Train:
    def __init__(self, seq_set, structure_set, seq_set_test = None, structure_set_test = None):
        self.seq_set = seq_set
        self.seq_length_set = []
        for x in self.seq_set:
            self.seq_length_set.append(len(x))
        self.structure_set = structure_set
        self.seq_set_test = seq_set_test
        self.structure_set_test = structure_set_test

    def hash_for_BP(self, i, j, N):
        n = j-i
        return int(n * (N + N - n +1)/2 + i)

    def train_engine(self, seq_set, true_structure_set):
        # model = Recursive.Recursive_net()
        start_iter = time()
        # model = self.model
        # optimizer = optimizers.Adam()
        # optimizer = optimizers.SGD()
        # optimizer.setup(model)
        for seq, true_structure in zip(seq_set, true_structure_set):
            inference = Inference.Inference(seq)
            predicted_BP = inference.ComputeInsideOutside(self.model)
            predicted_structure = inference.ComputePosterior(predicted_BP)

            i = 0
            for predicted_pair in predicted_structure:
                j = 0
                for true_pair in true_structure:
                    if (predicted_pair == true_pair).all():
                        np.delete(predicted_structure,i,0)
                        np.delete(true_structure,j,0)
                        break
                    j+=1
                i+=1

            loss = 0
            t = Variable(np.array([[0]], dtype=np.float32))
            t2 = Variable(np.array([[1]], dtype=np.float32))
            #backprop
            for predicted_pair in predicted_structure:
                y = predicted_BP[self.hash_for_BP(predicted_pair[0], predicted_pair[1], len(seq))]
                loss += F.mean_squared_error(y, t)
            for true_pair in true_structure:
                y = predicted_BP[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))]
                loss += F.mean_squared_error(y, t2)
            self.model.zerograds()
            loss.backward()
            self.optimizer.update()

        return self.model


    def wrapper_train(self,batch):
        seq_set, structure_set = zip(*batch)
        return self.train_engine(seq_set, structure_set)

    def train(self):
        #define model
        self.model = Recursive.Recursive_net()
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        list_for_shuffle = list(zip(self.seq_set, self.structure_set))
        random.shuffle(list_for_shuffle)

        for ite in range(iters_num):
            print('ite = ' +  str(ite))
            start_iter = time()

            if BATCH:
                #sort dataset
                list_for_sort = list(zip(self.seq_length_set, self.seq_set, self.structure_set))
                list_for_sort.sort(key=itemgetter(0))
                # list_for_batches = [list_for_sort[i:i+batch_num] for i in range(0,len(list_for_sort),batch_num)]
                i=0
                list_for_batches = []
                while i < len(list_for_sort):
                    a = math.ceil((list_for_sort[i][0]/60)*(list_for_sort[i][0]/60))
                    batch_num_sort =  a if a < maximum_slots else maximum_slots
                    list_for_batches.append(list_for_sort[i:i+batch_num_sort])
                    i = i + batch_num_sort

                random.shuffle(list_for_batches)
                # for seq, true_structure in zip(tqdm(seq_set), structure_set):
                start = time()
                for batch in list_for_batches:
                    start_batch =time()
                    seq_length_set, seq_set, structure_set = zip(*batch)
                    p= Pool(len(batch))
                    # p= Pool(multi.cpu_count())
                    result = p.map(self.wrapper_train, batch)
                    p.close()
                    loss = sum(result)
                    # for seq, true_structure in zip(seq_set, structure_set):

                    model.zerograds()
                    loss.backward()
                    optimizer.update()

            elif Iterative_Parameter_Mixture:
                # shuffle dataset
                # list_for_shuffle = list(zip(self.seq_set, self.structure_set))
                # random.shuffle(list_for_shuffle)
                batch_contents = math.ceil(len(list_for_shuffle)/batch_num)
                list_for_batches = [list_for_shuffle[i:i+batch_contents] for i in range(0,len(list_for_shuffle),batch_contents)]

                p= Pool(len(list_for_batches))
                model_set = p.map(self.wrapper_train, list_for_batches)
                p.close()

                # model_set = Parallel(n_jobs=-1)([delayed(self.wrapper_train)(batch) for batch in list_for_batches])

                l1_W = []
                l1_b = []
                l2_W = []
                l2_b = []
                L1_W = []
                L1_b = []
                L2_W = []
                L2_b = []

                for i in range(len(list_for_batches)):
                    l1_W.append(model_set[i].l1.W.data)
                    l1_b.append(model_set[i].l1.b.data)
                    l2_W.append(model_set[i].l2.W.data)
                    l2_b.append(model_set[i].l2.b.data)
                    L1_W.append(model_set[i].L1.W.data)
                    L1_b.append(model_set[i].L1.b.data)
                    L2_W.append(model_set[i].L2.W.data)
                    L2_b.append(model_set[i].L2.b.data)

                self.model.l1.W.data = sum(l1_W)/len(list_for_batches)
                self.model.l1.b.data = sum(l1_b)/len(list_for_batches)
                self.model.l2.W.data = sum(l2_W)/len(list_for_batches)
                self.model.l2.b.data = sum(l2_b)/len(list_for_batches)
                self.model.L1.W.data = sum(L1_W)/len(list_for_batches)
                self.model.L1.b.data = sum(L1_b)/len(list_for_batches)
                self.model.L2.W.data = sum(L2_W)/len(list_for_batches)
                self.model.L2.b.data = sum(L2_b)/len(list_for_batches)




            #SGD
            else:
                # shuffle dataset
                list_for_shuffle = list(zip(self.seq_set, self.structure_set))
                random.shuffle(list_for_shuffle)
                seq_set, structure_set = zip(*list_for_shuffle)
                # self.model = self.train_engine(seq_set, structure_set)
                self.train_engine(seq_set, structure_set)
                # print(str(i)+"/"+str(len(self.seq_set)))



            print('ite'+str(ite)+': it cost '+str(time() - start_iter)+'sec')
            #save model
            serializers.save_npz("NEURALfold_params.data" + str(ite), self.model)
            serializers.save_npz("NEURALfold_params.data", self.model)
            # serializers.save_npz('my.state'+ str(ite), optimizer)

            #TEST
            # if (self.seq_set_test is not None and i == iters_num-1):
            if (self.seq_set_test is not None):
                print("start testing...")
                test = Test.Test(self.seq_set_test)
                predicted_structure_set = test.test()
                evaluate = Evaluate.Evaluate(predicted_structure_set, self.structure_set_test)
                Sensitivity, PPV, F_value = evaluate.getscore()
                if ite == 0:
                    file = open('result.txt', 'w')
                else:
                    file = open('result.txt', 'a')

                result = ['Sensitivity=', str(round(Sensitivity,5)),' ', 'PPV=', str(round(PPV,5)),' ','F_value=', str(round(F_value,5)),'\n']
                file.writelines(result)
                print(Sensitivity, PPV, F_value)

            # print('ite'+str(ite)+': it cost '+str(time() - start)+'sec')

        return self.model
