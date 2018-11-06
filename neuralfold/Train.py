import numpy as np
from . import Config
from . import Recursive
import chainer.links as L
import chainer.functions as F
from . import Inference
from time import time
from .import Test
from . import Evaluate
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
import random
import os
from operator import itemgetter
import math
from . import SStruct
import pickle


class Train:
    def __init__(self, args):
        sstruct = SStruct.SStruct(args.train_file,train=False)
        if args.bpseq:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_BPseq()

        else:
            self.name_set, self.seq_set, self.structure_set = sstruct.load_FASTA()

        if args.test_file:
            sstruct = SStruct.SStruct(args.test_file)
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

        if args.init_parameters:
            serializers.load_npz(args.init_parameters, self.model)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        if args.init_optimizers:
            serializers.load_npz(args.init_optimizers, self.optimizer)

        self.feature = args.feature
        self.pos_margin = args.positive_margin
        self.neg_margin = args.negative_margin
        self.fully_learn = args.fully_learn
        self.ipknot = args.ipknot
        self.test_file = args.test_file
        if self.ipknot:
            if args.gamma:
                self.gamma = args.gamma
            else:
                self.gamma = [3.0,3.0]
        else:
            if args.gamma:
                self.gamma = args.gamma[0]
            else:
                self.gamma = 1.0
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
        # print(seq_set)
        # print(true_structure_set)
        for name, seq, true_structure in zip(name_set, seq_set, true_structure_set):
            print(name, len(seq), 'bp')
            print(seq)
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

            # print(' BP : '+str(time() - start_BP)+'sec')

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

                # Use nussinov regardless of structure when learning
                # self.ipknot = None

                predicted_structure = inference.ComputePosterior(predicted_BP.data, self.ipknot,
                                                                 gamma=self.gamma, margin=margin)

                predicted_score = inference.calculate_score(predicted_BP, self.ipknot, predicted_structure,
                                                            prediction = True, gamma=self.gamma, margin=margin)
                true_score = inference.calculate_score(predicted_BP, self.ipknot, true_structure,prediction=False, gamma=self.gamma)
                loss = predicted_score - true_score

            # print(inference.seq)
            # print(inference.dot_parentheis(predicted_structure))
            # print(inference.dot_parentheis(true_structure))
            # print(' structure : '+str(time() - start_structure)+'sec')

            start_backward = time()
            self.model.zerograds()
            loss.backward()
            # print(' backward : '+str(time() - start_backward)+'sec')
            start_update = time()
            self.optimizer.update()
            # print(' update : '+str(time() - start_update)+'sec')

        return self.model


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

            # print('ite'+str(ite)+': it cost '+str(time() - start_iter)+'sec')

            #save model
            self.save_model(self.args.parameters + str(ite))
            serializers.save_npz(self.args.optimizers + str(ite), self.optimizer)

            #TEST
            if (self.test_file is not None):
                predicted_structure_set = []
                print("start testing...")
                for seq in self.seq_set_test:
                    inference = Inference.Inference(seq,self.feature)
                    if self.args.learning_model == "recursive":
                        predicted_BP = inference.ComputeInsideOutside(self.model)
                    elif self.args.learning_model == "deepnet":
                        predicted_BP = inference.ComputeNeighbor(self.model)
                    else:
                        print("unexpected network")

                    predicted_structure = inference.ComputePosterior(predicted_BP.data, self.ipknot, self.gamma)
                    if len(predicted_structure) == 0:
                        continue
                    if self.ipknot:
                        predicted_structure = predicted_structure[0] + predicted_structure[1]
                    predicted_structure_set.append(predicted_structure)

                # print(predicted_structure_set[0])
                # print(self.structure_set_test[0])

                evaluate = Evaluate.Evaluate(predicted_structure_set, self.structure_set_test)
                Sensitivity, PPV, F_value = evaluate.get_score()

                file = open('result.txt', 'a')
                result = ['Sensitivity=', str(round(Sensitivity,5)),' ',
                          'PPV=', str(round(PPV,5)),' ',
                          'F_value=', str(round(F_value,5)),' ',str(self.gamma),'\n']
                file.writelines(result)
                file.close()
                print(Sensitivity, PPV, F_value)

            # print('ite'+str(ite)+': it cost '+str(time() - start)+'sec')

        self.save_model(self.args.parameters)

        return self.model

    def save_model(self, f):
        with open(f+'.pickle', 'wb') as fp:
            if self.args.learning_model == "recursive":
                pickle.dump(["recursive",
                             self.model.hidden_insideoutside,
                             self.model.hidden2_insideoutside,
                             self.model.feature,
                             self.model.hidden_merge,
                             self.hidden2_merge], fp)

            elif self.args.learning_model == "deepnet":
                pickle.dump(["deepnet",
                             self.model.neighbor,
                             self.model.hidden1,
                             self.model.hidden2], fp)

            else:
                print("unexpected network")
                return

        serializers.save_npz(f+'.npz', self.model)

    @classmethod
    def add_args(cls, parser):
        import argparse
        # add subparser for training
        parser_training = parser.add_parser('train', help='training RNA secondary structures')
        parser_training.add_argument('train_file', help = 'FASTA or BPseq file for training', nargs='+',
                                    type=argparse.FileType('r'))
        parser_training.add_argument('-t','--test_file', help = 'FASTA file for test', nargs='+',
                                    type=argparse.FileType('r'))
        parser_training.add_argument('--init-parameters', help = 'Initial parameter file',
                                    type=str)
        parser_training.add_argument('-p','--parameters', help = 'Output parameter file',
                                    type=str, default="NEURALfold_parameters")
        parser_training.add_argument('--init-optimizers', help = 'Initial optimizer state file',
                                    # type=argparse.FileType('r'))
                                    type=str)
        parser_training.add_argument('-o','--optimizers', help = 'Optimizer state file',
                                    type=str, default="state.npz")
        parser_training.add_argument('-i','--iteration', help = 'number of iteration',
                                    type=int, default=1)
        parser_training.add_argument('-bp','--bpseq', help = 'use bpseq',
                                    action = 'store_true')

        # neural networks architecture
        parser_training.add_argument('-l','--learning_model',
                                    help = 'learning_model',
                                    type=str, default="deepnet")

        parser_training.add_argument('-H1','--hidden_insideoutside',
                                    help = 'hidden layer nodes for inside outside',
                                    type=int, default=80)
        parser_training.add_argument('-H2','--hidden2_insideoutside',
                                    help = 'hidden layer nodes2 for inside outside',
                                    type=int)
        parser_training.add_argument('-h1','--hidden_merge',
                                    help = 'hidden layer nodes for merge phase',
                                    type=int, default=80)
        parser_training.add_argument('-h2','--hidden2_merge',
                                    help = 'hidden layer nodes2 for merge phase',
                                    type=int)
        parser_training.add_argument('-f','--feature',
                                    help = 'feature length',
                                    type=int, default=40)

        parser_training.add_argument('-hn1','--hidden1',
                                    help = 'hidden layer nodes for neighbor model',
                                    type=int, default=200)
        parser_training.add_argument('-hn2','--hidden2',
                                    help = 'hidden layer nodes2 for neighbor model',
                                    type=int, default=50)
        parser_training.add_argument('-n','--neighbor',
                                    help = 'length of neighbor bases to see',
                                    type=int, default=40)

        # training option
        parser_training.add_argument('-g','--gamma',
                                    help = 'balance between the sensitivity and specificity ',
                                    type=float, action='append')
        parser_training.add_argument('-m','--positive-margin',
                                    help = 'margin for positives',
                                    type=float, default=0.2)
        parser_training.add_argument('--negative-margin',
                                    help = 'margin for negatives',
                                    type=float, default=0.2)
        parser_training.add_argument('-fu','--fully_learn',
                                    help = 'calculate loss for all canonical pair',
                                    action = 'store_true')
        parser_training.add_argument('-ip','--ipknot',
                                    help = 'predict pseudoknotted secaondary structure',
                                    action = 'store_true')

        parser_training.set_defaults(func = lambda args: Train(args).train())