import numpy as np
import Config
import Recursive
import Deepnet
import chainer.links as L
import chainer.functions as F
import Inference
from time import time
import Test
import Evaluate
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
import random
import os
from operator import itemgetter
import math
import SStruct


batch_num = Config.batch_num
maximum_slots = Config.maximum_slots
BATCH = Config.BATCH
Iterative_Parameter_Mixture = Config.Iterative_Parameter_Mixture

class Train:
    def __init__(self, args):
        sstruct = SStruct.SStruct(args.train_file, args.small_structure)
        if args.bpseq:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_BPseq()
        else:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_FASTA()

        if args.test_file:
            sstruct = SStruct.SStruct(args.test_file, args.small_structure)
            if args.bpseq:
                self.name_set_test, self.seq_set_test, self.structure_set_test = sstruct.load_BPseq()
            else:
                self.name_set_test, self.seq_set_test, self.structure_set_test = sstruct.load_FASTA()
        else:
            self.seq_set_test = None

        self.iters_num = args.iteration
        # self.seq_length_set = []
        # for x in self.seq_set:
        #     self.seq_length_set.append(len(x))

        if args.learning_model == "recursive":
            self.model = Recursive.Recursive_net(args.hidden_insideoutside, args.hidden2_insideoutside, args.feature,
                                                 args.hidden_merge, args.hidden2_merge)
        elif args.learning_model == "deepnet":
            self.model = Deepnet.Deepnet(args.neighbor, args.hidden1, args.hidden2)
        else:
            print("unexpected network")

        if args.Parameters:
            serializers.load_npz(args.Parameters.name, self.model)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        if args.Optimizers:
            serializers.load_npz(args.Optimizers.name, self.optimizer)

        self.feature = args.feature
        self.pos_margin = args.margin
        self.neg_margin = args.margin
        self.fully_learn = args.fully_learn
        self.ipknot = args.ipknot
        self.test_file = args.test_file
        self.gamma = args.gamma
        if self.ipknot:
            self.gamma = (self.gamma, self.gamma)
        self.args = args


    def hash_for_BP(self, i, j, N):
        n = j-i
        return int(n * (N + N - n +1)/2 + i)

    def train_engine(self, name_set, seq_set, true_structure_set):
        # model = Recursive.Recursive_net()
        start_iter = time()
        # model = self.model
        # optimizer = optimizers.Adam()
        # optimizer = optimizers.SGD()
        # optimizer.setup(model)
        step = 0
        for name, seq, true_structure in zip(name_set, seq_set, true_structure_set):
            print(name, len(seq), 'bp')
            # print(seq)
            # print(true_structure)
            step += 1
            start_BP = time()
            inference = Inference.Inference(seq, self.feature)
            if self.args.learning_model == "recursive":
                predicted_BP = inference.ComputeInsideOutside(self.model)
            elif self.args.learning_model == "deepnet":
                predicted_BP = inference.ComputeNeighbor(self.model)
            else:
                print("unexpected network")

            print(' BP : '+str(time() - start_BP)+'sec')

            if self.fully_learn:
                length = len(seq)
                num = int((length*length-length)/2+length)
                true_structure_matrix = Variable(np.zeros((num, 1), dtype=np.float32))

                for true_pair in true_structure:
                    true_structure_matrix[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))] = Variable(np.array([1], dtype=np.float32))
                loss = F.mean_squared_error(predicted_BP.reshape(num, 1), true_structure_matrix)

            else:
                start_structure = time()
                length = len(seq)

                margin = np.zeros((length,length), dtype=np.float32)
                margin += self.neg_margin
                for true_pair in true_structure:
                    margin[true_pair[0], true_pair[1]] -= self.pos_margin + self.neg_margin

                predicted_structure = inference.ComputePosterior(predicted_BP.data, self.ipknot,
                                                                 gamma=self.gamma, margin=margin)
                predicted_score = inference.calculate_score(predicted_BP, predicted_structure,
                                                            gamma=self.gamma, margin=margin)
                true_score = inference.calculate_score(predicted_BP, true_structure, gamma=self.gamma)
                loss = predicted_score - true_score

            print(' structure : '+str(time() - start_structure)+'sec')

            start_backward = time()
            self.model.zerograds()
            loss.backward()
            print(' backward : '+str(time() - start_backward)+'sec')
            start_update = time()
            self.optimizer.update()
            print(' update : '+str(time() - start_update)+'sec')

        return self.model


    def wrapper_train(self, batch):
        name_set, seq_set, structure_set = zip(*batch)
        return self.train_engine(name_set, seq_set, structure_set)

    def train(self):
        #define model
        # self.model = Recursive.Recursive_net()
        # self.optimizer = optimizers.Adam()
        # self.optimizer.setup(self.model)

        for ite in range(self.iters_num):
            print('ite = ' +  str(ite))
            start_iter = time()

            list_for_shuffle = list(zip(self.name_set, self.seq_set, self.structure_set))
            random.shuffle(list_for_shuffle)
            name_set, seq_set, structure_set = zip(*list_for_shuffle)
            self.train_engine(name_set, seq_set, structure_set)

            # print(str(i)+"/"+str(len(self.seq_set)))

            print('ite'+str(ite)+': it cost '+str(time() - start_iter)+'sec')
            #save model
            serializers.save_npz("NEURALfold_params.data" + str(ite), self.model)
            serializers.save_npz("NEURALfold_params.data", self.model)
            serializers.save_npz('my.state'+ str(ite), self.optimizer)
            serializers.save_npz('my.state', self.optimizer)

            #TEST
            if (self.test_file is not None):
                predicted_structure_set = []
                print("start testing...")
                for seq in self.seq_set_test:
                    pinference = Inference.Inference(seq,self.feature)
                    if self.args.learning_model == "recursive":
                        predicted_BP = inference.ComputeInsideOutside(self.model)
                    elif self.args.learning_model == "deepnet":
                        predicted_BP = inference.ComputeNeighbor(self.model)
                    else:
                        print("unexpected network")

                    predicted_structure = inference.ComputePosterior(predicted_BP, self.ipknot, self.gamma)
                    if len(predicted_structure) == 0:
                        continue
                    predicted_structure_set.append(predicted_structure)

                    evaluate = Evaluate.Evaluate(predicted_structure_set, self.structure_set_test)
                    Sensitivity, PPV, F_value = evaluate.getscore()

                    file = open('result.txt', 'a')
                    result = ['Sensitivity=', str(round(Sensitivity,5)),' ',
                              'PPV=', str(round(PPV,5)),' ',
                              'F_value=', str(round(F_value,5)),' ',str(self.gamma),'\n']
                    file.writelines(result)
                    file.close()
                print(Sensitivity, PPV, F_value)

            # print('ite'+str(ite)+': it cost '+str(time() - start)+'sec')

        return self.model
