import chainer
import chainer.functions as F
import numpy as np
from chainer import Variable, reporter
from chainer.cuda import get_array_module, to_cpu
from joblib import Parallel, delayed

from . import evaluate


class StructuredLoss(chainer.Chain):
    def __init__(self, model, decoder, positive_margin, negative_margin,
                compute_accuracy=False, verbose=False):
        super(StructuredLoss, self).__init__()

        with self.init_scope():
            self.model = model

        self.decoder = decoder
        self.pos_margin = positive_margin
        self.neg_margin = negative_margin
        self.compute_accuracy = compute_accuracy
        self.verbose = verbose


    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('Options for structured loss function')
        group.add_argument('-m','--positive-margin',
                            help='margin for positives',
                            type=float, default=0.2)
        group.add_argument('--negative-margin',
                            help='margin for negatives',
                            type=float, default=0.2)


    @classmethod    
    def parse_args(cls, args):
        hyper_params = ('positive_margin', 'negative_margin')
        return {p: getattr(args, p, None) for p in hyper_params if getattr(args, p, None) is not None}

    
    def __call__(self, name_set, seq_set, true_structure_set):
        def func(decoder, id, seq, bpp, true_structure=None, pos_margin=0., neg_margin=0.):
            N = len(seq)
            margin = None
            if true_structure is not None:
                margin = np.full((N, N), neg_margin, dtype=np.float32)
                for i, j in true_structure:
                    margin[i, j] -= pos_margin + neg_margin
            return (id, decoder.decode(seq, bpp[0:N, 0:N], margin=margin), margin)

        B = len(seq_set)
        loss = 0
        bpp = self.model.compute_bpp(seq_set)
        bpp_array = to_cpu(bpp.array)

        jobs = [
            delayed(func)(self.decoder, k, seq, bpp_array[k], 
                    true_structure, self.pos_margin, self.neg_margin)
                for k, (seq, true_structure) in enumerate(zip(seq_set, true_structure_set))
        ]
        r = sorted(Parallel(n_jobs=-1)(jobs))
        predicted_structure_set = [ s[1] for s in r ]
        margin_set = [ s[2] for s in r ]

        for k, (name, seq, true_structure, predicted_structure, margin) in enumerate(zip(name_set, seq_set, true_structure_set, predicted_structure_set, margin_set)):
            predicted_score = self.decoder.calc_score(seq, bpp[k], pair=predicted_structure, margin=margin)
            predicted_score += self.pos_margin * len(true_structure)
            true_score = self.decoder.calc_score(seq, bpp[k], pair=true_structure)
            loss += (predicted_score - true_score) / len(seq)
            if self.verbose:
                print(name)
                print(seq)
                print(self.decoder.dot_parenthesis(seq, true_structure))                
                print(self.decoder.dot_parenthesis(seq, predicted_structure))
                print('predicted score = {:.3f}, true score = {:.3f}'
                        .format(to_cpu(predicted_score.data[0]), to_cpu(true_score.data[0])))
                print()

        loss = loss[0] / B
        reporter.report({'loss': loss}, self)

        if self.compute_accuracy:
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