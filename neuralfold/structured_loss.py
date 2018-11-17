import chainer
import numpy as np
from chainer.cuda import to_cpu
from chainer import reporter, Variable
import chainer.functions as F
from . import evaluate

class StructuredLoss(chainer.Chain):
    def __init__(self, model, decoder, pos_margin, neg_margin, compute_accuracy=False):
        super(StructuredLoss, self).__init__()

        with self.init_scope():
            self.model = model

        self.decoder = decoder
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.compute_accuracy = compute_accuracy


    def __call__(self, name_set, seq_set, true_structure_set):
        B = len(seq_set)
        loss = 0
        pred_structure_set = []
        predicted_BP = self.model.compute_bpp(seq_set)
        #for k, (name, seq, true_structure) in enumerate(zip(name_set, seq_set, true_structure_set)):
        for k, (seq, true_structure) in enumerate(zip(seq_set, true_structure_set)):
            N = len(seq)
            #print(name, N, 'bp')
            margin = np.full((N, N), self.neg_margin, dtype=np.float32)
            for i, j in true_structure:
                margin[i, j] -= self.pos_margin + self.neg_margin

            predicted_structure = self.decoder.decode(seq, to_cpu(predicted_BP[k].array[0:N, 0:N]), margin=margin)
            predicted_score = self.decoder.calc_score(seq, predicted_BP[k], pair=predicted_structure, margin=margin)
            true_score = self.decoder.calc_score(seq, predicted_BP[k], pair=true_structure)
            loss += predicted_score - true_score

            if self.compute_accuracy:
                pred_structure = self.decoder.decode(seq, to_cpu(predicted_BP[k].array[0:N, 0:N]))
                pred_structure_set.append(pred_structure)

        loss = loss[0] / B
        reporter.report({'loss': loss}, self)

        if self.compute_accuracy:
            sen, ppv, f_val = evaluate.get_score(true_structure_set, pred_structure_set, 
                                        global_average=False)
            reporter.report({'sen': sen, 'ppv': ppv, 'f_val': f_val}, self)
        
        return loss
