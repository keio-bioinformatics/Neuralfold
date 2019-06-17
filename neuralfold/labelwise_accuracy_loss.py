import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable, reporter
from chainer.cuda import get_array_module, to_cpu
from joblib import Parallel, delayed

from . import evaluate


class LabelwiseAccuracyLoss(chainer.Chain):
    def __init__(self, model, decoder, compute_accuracy=False, verbose=False):
        super(LabelwiseAccuracyLoss, self).__init__()

        with self.init_scope():
            self.model = model

        self.decoder = decoder
        self.compute_accuracy = compute_accuracy
        self.verbose = verbose


    @classmethod
    def add_args(cls, parser):
        # group = parser.add_argument_group('Options for structured loss function')
        pass


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ()
        return {p: getattr(args, p, None) for p in hyper_params if getattr(args, p, None) is not None}

    
    def __call__(self, name_set, seq_set, true_structure_set):

        def flattern(paren_list):
            if len(paren_list) == 0 or len(paren_list[0]) != 0 and type(paren_list[0][0]) is int:
                return paren_list
            else:
                ret = []
                for p in paren_list:
                    ret += p
                return ret

        def func(decoder, id, seq, bpp):
            N = len(seq)
            return (id, decoder.decode(seq, bpp[0:N, 0:N]))

        B = len(seq_set)
        loss = 0
        bpp = self.model.compute_bpp(seq_set)
        bpp_array = to_cpu(bpp.array)

        jobs = [
            delayed(func)(self.decoder, k, seq, bpp_array[k])
                for k, seq in enumerate(seq_set)
        ]
        r = sorted(Parallel(n_jobs=-1)(jobs))
        predicted_structure_set = [ s[1] for s in r ]
        gamma = min(self.decoder.gamma) if isinstance(self.decoder.gamma, list) else self.decoder.gamma

        for k, (name, seq, true_structure, predicted_structure) in enumerate(zip(name_set, seq_set, true_structure_set, predicted_structure_set)):
            ref_str = set(flattern(true_structure))
            pred_str = set(flattern(predicted_structure))
            tp = ref_str & pred_str
            fp = pred_str - tp
            fn = ref_str - tp
            n_tp = n_fp = n_fn = 0
            for i, j in tp:
                n_tp += F.sigmoid(  (gamma+1) * bpp[k][i, j] -1)
            for i, j in fp:
                n_fp += F.sigmoid(  (gamma+1) * bpp[k][i, j] -1)
            for i, j in fn:
                n_fn += F.sigmoid(-((gamma+1) * bpp[k][i, j] -1))
            ppv = n_tp / (n_tp + n_fp) if (n_tp + n_fp).data != 0 else 0
            sen = n_tp / (n_tp + n_fn) if (n_tp + n_fn).data != 0 else 0
            f_val = 2 * (sen * ppv) / (sen + ppv) if (sen + ppv).data != 0 else Variable(bpp.xp.zeros((), dtype=np.float32))
            loss += 1 - f_val

            if self.verbose:
                print(name)
                print(seq)
                predicted_score = self.decoder.calc_score(seq, bpp_array[k], pair=predicted_structure)
                true_score = self.decoder.calc_score(seq, bpp_array[k], pair=true_structure)
                print(self.decoder.dot_parenthesis(seq, true_structure))                
                print(self.decoder.dot_parenthesis(seq, predicted_structure))
                print('predicted score = {:.3f}, true score = {:.3f}'
                        .format(to_cpu(predicted_score.data[0]), to_cpu(true_score.data[0])))
                print()

        loss = loss / B
        reporter.report({'loss': loss}, self)

        if self.compute_accuracy:
            sen, ppv, f_val = evaluate.get_score(true_structure_set, predicted_structure_set ,
                                        global_average=False)
            reporter.report({'sen': sen, 'ppv': ppv, 'f_val': f_val}, self)
        
        return loss