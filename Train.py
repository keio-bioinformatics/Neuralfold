import numpy as np
import Config
import Recursive
import chainer.links as L
import chainer.functions as F
import Inference
from tqdm import tqdm
from time import time
#import chainer
from chainer import optimizers, Chain, Variable, cuda, optimizer, serializers
iters_num = Config.iters_num

class Train:
    def __init__(self, seq_set, structure_set):
        self.seq_set = seq_set
        self.structure_set = structure_set

    def train(self):
        #define model
        model = Recursive.Recursive_net()
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        for ite in range(iters_num):
            start = time()
            print('ite = ' +  str(ite))
            for seq, true_structure in zip(tqdm(self.seq_set), self.structure_set):

                inference = Inference.Inference(seq)
                predicted_BP = inference.ComputeInsideOutside(model)
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

                model.zerograds()
                loss.backward()
                optimizer.update()
            print('it cost {}sec'.format(time() - start))

        return model
