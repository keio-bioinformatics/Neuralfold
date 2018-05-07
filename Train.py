import numpy as np
import Config
import Recursive
import Deepnet
import chainer.links as L
import chainer.functions as F
import Inference
# from tqdm import tqdm
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
# from joblib import Parallel, delayed
import SStruct


batch_num = Config.batch_num
maximum_slots = Config.maximum_slots
BATCH = Config.BATCH
Iterative_Parameter_Mixture = Config.Iterative_Parameter_Mixture

class Train:
    # def __init__(self, seq_set, structure_set, seq_set_test = None, structure_set_test = None):
    #     self.seq_set = seq_set
    #     self.seq_length_set = []
    #     for x in self.seq_set:
    #         self.seq_length_set.append(len(x))
    #     self.structure_set = structure_set
    #     self.seq_set_test = seq_set_test
    #     self.structure_set_test = structure_set_test
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

        # self.model = Recursive.Recursive_net(args)
        if args.learning_model == "recursive":
            self.model = Recursive.Recursive_net(args.hidden_insideoutside,args.hidden2_insideoutside,args.feature,
                                                 args.hidden_marge,args.hidden2_marge, args.activation_function)
        elif args.learning_model == "deepnet":
            self.model = Deepnet.Deepnet(args.hidden1, args.hidden2, args.hidden3, args.activation_function)
        else:
            print("unexpected network")

        if args.Parameters:
            serializers.load_npz(args.Parameters.name, self.model)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        if args.Optimizers:
            serializers.load_npz(args.Optimizers.name, self.optimizer)

        self.activation_function = args.activation_function
        self.feature = args.feature
        self.max_margin = args.max_margin
        self.unpair_score = args.unpair_score
        self.unpair_weight = args.unpair_weight
        self.neighbor = args.neighbor
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

    def Useloss(self,predicted_BP, seq, true_structure):
        margin = 0.2
        if self.activation_function == "softmax":
            predicted_BP[:,:,1].data += margin
            predicted_BP[:,:,0].data -= margin
        elif self.activation_function == "sigmoid":
            predicted_BP.data += margin
        else:
            print("enexpected function")

        for true_pair in true_structure:
            if self.activation_function == "softmax":
                predicted_BP[self.hash_for_BP(true_pair[0], true_pair[1], len(seq)),0,1].data -= 2*margin
                predicted_BP[self.hash_for_BP(true_pair[0], true_pair[1], len(seq)),0,0].data += 2*margin
            elif self.activation_function == "sigmoid":
                predicted_BP[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))].data -= 2*margin
            else:
                print("enexpected function")

        return predicted_BP

    def calculate_loss(self, predicted_structure, true_structure, predicted_BP_UP, seq, pair_unpair):
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
        if self.activation_function == "softmax":
            t = Variable(np.array([0], dtype=np.int32))
            t2 = Variable(np.array([1], dtype=np.int32))

        elif self.activation_function == "sigmoid":
            t = Variable(np.array([[0]], dtype=np.float32))
            if pair_unpair == "pair":
                t2 = Variable(np.array([[1]], dtype=np.float32))
            else:
                t2 = Variable(np.array([[1]], dtype=np.float32))
        else:
            print("enexpected function")
        #backprop
        count = 0
        for predicted_pair in predicted_structure:
            count += 1
            y = predicted_BP_UP[self.hash_for_BP(predicted_pair[0], predicted_pair[1], len(seq))]
            if self.activation_function == "softmax":
                loss += F.softmax_cross_entropy(y, t)
            elif self.activation_function == "sigmoid":
                loss += F.mean_squared_error(y, t)
            else:
                print("enexpected function")

        for true_pair in true_structure:
            count += 1
            y = predicted_BP_UP[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))]
            if self.activation_function == "softmax":
                loss += F.softmax_cross_entropy(y, t2)
            elif self.activation_function == "sigmoid":
                loss += F.mean_squared_error(y, t2)
            else:
                print("enexpected function")

        return loss


    def train_engine(self, seq_set, true_structure_set):
        # model = Recursive.Recursive_net()
        start_iter = time()
        # model = self.model
        # optimizer = optimizers.Adam()
        # optimizer = optimizers.SGD()
        # optimizer.setup(model)
        step=0
        for seq, true_structure in zip(seq_set, true_structure_set):
            print("step=",str(step))
            # if (seq=="GGGAAACGGGCAGGCGGCGGCGACCGCCGAAACAACCGC"):
            #     continue
            # print(seq)
            # print(true_structure)
            step +=1
            start_BP = time()
            inference = Inference.Inference(seq,self.feature, self.activation_function,self.unpair_weight)
            if self.args.learning_model == "recursive":
                predicted_BP = inference.ComputeInsideOutside(self.model)
            elif self.args.learning_model == "deepnet":
                predicted_BP, predicted_UP_left, predicted_UP_right = inference.ComputeNeighbor(self.model, self.neighbor)
            else:
                print("unexpected network")

            # predicted_BP = inference.ComputeInsideOutside(self.model)

            print(' BP : '+str(time() - start_BP)+'sec')
            if self.max_margin:
                predicted_BP = self.Useloss(predicted_BP, seq, true_structure)


            if self.fully_learn:
                length = len(seq)
                num = int((length*length-length)/2+length)
                true_structure_matrix = Variable(np.zeros((num, 1), dtype=np.float32))
                loss = 0
                # if self.activation_function == "softmax":
                #     t = Variable(np.array([0], dtype=np.int32))
                #     t2 = Variable(np.array([1], dtype=np.int32))
                # elif self.activation_function == "sigmoid":
                #     t = Variable(np.array([[0]], dtype=np.float32))
                #     t2 = Variable(np.array([[1]], dtype=np.float32))
                # else:
                #     print("enexpected function")
                #
                # for true_pair in true_structure:
                #     y = predicted_BP[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))]
                #     if self.activation_function == "softmax":
                #         loss += F.softmax_cross_entropy(y, t2)
                #     elif self.activation_function == "sigmoid":
                #         loss += F.mean_squared_error(y, t2)
                #     else:
                #         print("enexpected function")
                # loss = loss*10

                for true_pair in true_structure:
                    true_structure_matrix[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))].data = Variable(np.array([1], dtype=np.float32))
                loss += F.mean_squared_error(predicted_BP.reshape(num, 1), true_structure_matrix)

            elif self.unpair_weight:
                # predict structure
                predicted_structure, predicted_unpair_left, predicted_unpair_right = inference.ComputePosterior_with_unpair(predicted_BP, predicted_UP_left, predicted_UP_right)

                # create true structure BPP
                length = len(seq)
                num = int((length*length-length)/2+length)
                true_structure_matrix = Variable(np.zeros((num, 1), dtype=np.float32))
                for true_pair in true_structure:
                    true_structure_matrix[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))].data = Variable(np.array([1], dtype=np.float32))

                ture_structure, true_unpair_left, true_unpair_right = inference.ComputePosterior_with_unpair(true_structure_matrix, None, None)
                predicted_UP_left = self.Useloss(predicted_UP_left, seq, true_unpair_left)
                predicted_UP_right = self.Useloss(predicted_UP_right, seq, true_unpair_right)
                loss = 0
                loss += self.calculate_loss(predicted_structure, true_structure, predicted_BP, seq, "pair")
                loss += self.calculate_loss(predicted_unpair_left, true_unpair_left, predicted_UP_left, seq, "unpair")
                loss += self.calculate_loss(predicted_unpair_right, true_unpair_right, predicted_UP_right, seq , "unpair")


            else:
                start_structure = time()
                length = len(seq)
                true_structure_matrix = np.zeros((length,length), dtype=np.float32)
                for true_pair in true_structure:
                    true_structure_matrix[true_pair[0], true_pair[1]] = self.unpair_score

                # predicted_structure = inference.ComputePosterior(predicted_BP, self.unpair_score, self.ipknot, self.gamma, "Train")
                # predicted_structure = inference.ComputePosterior(predicted_BP, self.unpair_score, self.ipknot, self.gamma, "Train",true_structure_matrix)
                predicted_structure = inference.ComputePosterior(predicted_BP, self.ipknot, self.gamma)
                if len(predicted_structure)==0:
                    continue
                print(' structure : '+str(time() - start_structure)+'sec')

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
                if self.activation_function == "softmax":
                    t = Variable(np.array([0], dtype=np.int32))
                    t2 = Variable(np.array([1], dtype=np.int32))
                elif self.activation_function == "sigmoid":
                    t = Variable(np.array([[0]], dtype=np.float32))
                    t2 = Variable(np.array([[1]], dtype=np.float32))
                else:
                    print("enexpected function")
                #backprop
                count = 0
                for predicted_pair in predicted_structure:
                    count += 1
                    y = predicted_BP[self.hash_for_BP(predicted_pair[0], predicted_pair[1], len(seq))]
                    if self.activation_function == "softmax":
                        loss += 2 * F.softmax_cross_entropy(y, t)
                    elif self.activation_function == "sigmoid":
                        loss += 2 * F.mean_squared_error(y, t)
                    else:
                        print("enexpected function")

                for true_pair in true_structure:
                    count += 1
                    y = predicted_BP[self.hash_for_BP(true_pair[0], true_pair[1], len(seq))]
                    if self.activation_function == "softmax":
                        loss += F.softmax_cross_entropy(y, t2)
                    elif self.activation_function == "sigmoid":
                        loss += F.mean_squared_error(y, t2)
                    else:
                        print("enexpected function")

                if self.args.count:
                    loss = count*loss

            start_backward = time()
            self.model.zerograds()
            loss.backward()
            # print(' backward : '+str(time() - start_backward)+'sec')
            start_update = time()
            self.optimizer.update()
            # print(' update : '+str(time() - start_update)+'sec')

        return self.model


    def wrapper_train(self,batch):
        seq_set, structure_set = zip(*batch)
        return self.train_engine(seq_set, structure_set)

    def train(self):
        #define model
        # self.model = Recursive.Recursive_net()
        # self.optimizer = optimizers.Adam()
        # self.optimizer.setup(self.model)

        list_for_shuffle = list(zip(self.seq_set, self.structure_set))
        random.shuffle(list_for_shuffle)

        for ite in range(self.iters_num):
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
                self.train_engine(seq_set, structure_set)

                # print(str(i)+"/"+str(len(self.seq_set)))



            print('ite'+str(ite)+': it cost '+str(time() - start_iter)+'sec')
            #save model
            serializers.save_npz("NEURALfold_params.data" + str(ite), self.model)
            serializers.save_npz("NEURALfold_params.data", self.model)
            serializers.save_npz('my.state'+ str(ite), self.optimizer)
            serializers.save_npz('my.state', self.optimizer)

            #TEST
            # if (self.seq_set_test is not None and ite == self.iters_num-1):
            # if (self.seq_set_test is not None and (ite % 10)==9 ):
            if (self.test_file is not None):
                predicted_structure_set = []
                print("start testing...")
                for seq in self.seq_set_test:
                    inference = Inference.Inference(seq,self.feature, self.activation_function, self.unpair_weight)
                    if self.args.learning_model == "recursive":
                        predicted_BP = inference.ComputeInsideOutside(self.model)
                    elif self.args.learning_model == "deepnet":
                        # predicted_BP = inference.ComputeNeighbor(self.model, self.neighbor)
                        predicted_BP, predicted_UP_left, predicted_UP_right = inference.ComputeNeighbor(self.model, self.neighbor)
                    else:
                        print("unexpected network")

                    if self.unpair_weight:
                        predicted_structure, predicted_unpair_left, predicted_unpair_right = inference.ComputePosterior_with_unpair(predicted_BP, predicted_UP_left, predicted_UP_right)
                        predicted_structure_set.append(predicted_structure)
                    else:
                        # predicted_structure=inference.ComputePosterior(predicted_BP, self.unpair_score, self.ipknot, self.gamma, "Test",np.zeros((len(seq),len(seq)), dtype=np.float32))
                        predicted_structure=inference.ComputePosterior(predicted_BP, self.ipknot, self.gamma, "Test")
                        # predicted_structure_set.append(inference.ComputePosterior(predicted_BP, self.unpair_score, self.ipknot,"Test"))
                        if len(predicted_structure)==0:
                            continue
                        predicted_structure_set.append(predicted_structure)

                    evaluate = Evaluate.Evaluate(predicted_structure_set, self.structure_set_test)
                    Sensitivity, PPV, F_value = evaluate.getscore()

                    file = open('result.txt', 'a')
                    result = ['Sensitivity=', str(round(Sensitivity,5)),' ', 'PPV=', str(round(PPV,5)),' ','F_value=', str(round(F_value,5)),' ',str(self.gamma),'\n']
                    file.writelines(result)
                    file.close()
                print(Sensitivity, PPV, F_value)

            # print('ite'+str(ite)+': it cost '+str(time() - start)+'sec')

        return self.model
