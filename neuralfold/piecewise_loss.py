import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable, reporter
from chainer.cuda import get_array_module, to_cpu
from joblib import Parallel, delayed

from . import evaluate


class PiecewiseLoss(chainer.Chain):
    def __init__(self, model, decoder, 
                compute_accuracy=False, verbose=False):
        super(PiecewiseLoss, self).__init__()

        with self.init_scope():
            self.model = model

        self.decoder = decoder
        self.compute_accuracy = compute_accuracy
        self.verbose = verbose


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for piece-wise loss function')


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ()
        return {p: getattr(args, p, None) for p in hyper_params if getattr(args, p, None) is not None}

    
    def __call__(self, name_set, seq_set, true_structure_set):
        def func(decoder, id, seq, bpp):
            N = len(seq)
            return (id, decoder.decode(seq, bpp[0:N, 0:N]))

        B = len(seq_set)
        loss = 0.
        bpp = self.model.compute_bpp(seq_set)

        for k, (seq, true_structure) in enumerate(zip(seq_set, true_structure_set)):
            N = len(seq)
            t = self.xp.zeros((N, N), dtype=np.int32)
            for i, j in true_structure:
                t[i, j] = 1
            l = 0
            for j in range(N):
                for i in range(j):
                    l += F.sigmoid_cross_entropy(bpp[k][i, j].reshape(1, 1), t[i, j].reshape(1, 1))
            loss += l / (N*(N-1)/2)

        loss = loss / B
        reporter.report({'loss': loss}, self)

        if self.compute_accuracy:
            bpp_array = to_cpu(bpp.array)
            jobs = [
                delayed(func)(self.decoder, k, seq, bpp_array[k],)
                    for k, seq in enumerate(seq_set)
            ]
            r = sorted(Parallel(n_jobs=-1)(jobs))
            pred_structure_set = [ s[1] for s in r ]
            sen, ppv, f_val = evaluate.get_score(true_structure_set, pred_structure_set, 
                                        global_average=False)
            reporter.report({'sen': sen, 'ppv': ppv, 'f_val': f_val}, self)
        
        return loss
