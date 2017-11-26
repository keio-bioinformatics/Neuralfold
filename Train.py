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


iters_num = Config.iters_num
TEST = Config.TEST
batch_num = Config.batch

class Train:
    def __init__(self, seq_set, structure_set, seq_set_test = None, structure_set_test = None):
        self.seq_set = seq_set
        self.structure_set = structure_set
        self.seq_set_test = seq_set_test
        self.structure_set_test = structure_set_test

    def train_engine(self, seq, true_structure):

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
            y = predicted_BP[predicted_pair[0]][predicted_pair[1]]
            loss += F.mean_squared_error(y, t)
        for true_pair in true_structure:
            y = predicted_BP[true_pair[0]][true_pair[1]]
            loss += F.mean_squared_error(y, t2)
        return loss


    def wrapper_train(self,args):
        return self.train_engine(*args)

    def train(self):
        #define self.model
        self.model = Recursive.Recursive_net()
        optimizer = optimizers.Adam()
        optimizer.setup(self.model)

        for ite in range(iters_num):
            print('ite = ' +  str(ite))
            #shuffle dataset
            list_for_shuffle = list(zip(self.seq_set, self.structure_set))
            random.shuffle(list_for_shuffle)
            list_for_batches = [list_for_shuffle[i:i+batch_num] for i in range(0,len(list_for_shuffle),batch_num)]

            # for seq, true_structure in zip(tqdm(seq_set), structure_set):

            start = time()
            for batch in list_for_batches:
                seq_set, structure_set = zip(*batch)
                # print(batch)
                p= Pool(80)
                # p= Pool(multi.cpu_count())
                result = p.map(self.wrapper_train, batch)
                p.close()
                loss = sum(result)
                # for seq, true_structure in zip(seq_set, structure_set):

                self.model.zerograds()
                loss.backward()
                optimizer.update()
                print('it cost {}sec'.format(time() - start))

            #save self.model
            serializers.save_npz("NEURALfold_params.data" + str(ite), self.model)
            serializers.save_npz("NEURALfold_params.data", self.model)
            serializers.save_npz('my.state'+ str(ite), optimizer)

            #TEST
            if (self.seq_set_test is not None):
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

            print('it cost {}sec'.format(time() - start))

        return self.model
