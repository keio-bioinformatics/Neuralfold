import chainer
import numpy as np
from chainer.cuda import to_cpu


class StructuredLoss(chainer.Chain):
    def __init__(self, model, decoder, pos_margin, neg_margin):
        super(StructuredLoss, self).__init__()

        with self.init_scope():
            self.model = model

        self.decoder = decoder
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin


    def __call__(self, name_set, seq_set, true_structure_set):
        loss = 0
        print(name_set)
        predicted_BP = self.model.compute_bpp(seq_set)
        for k, (name, seq, true_structure) in enumerate(zip(name_set, seq_set, true_structure_set)):
            N = len(seq)
            print(name, N, 'bp')
            margin = np.full((N, N), self.neg_margin, dtype=np.float32)
            for i, j in true_structure:
                margin[i, j] -= self.pos_margin + self.neg_margin

            predicted_structure = self.decoder.decode(to_cpu(predicted_BP[k].array[0:N, 0:N]), margin=margin)
            predicted_score = self.decoder.calc_score(predicted_BP[k], pair=predicted_structure, margin=margin)
            true_score = self.decoder.calc_score(predicted_BP[k], pair=true_structure)
            loss += predicted_score - true_score

        return loss
