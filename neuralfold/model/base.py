from chainer import Chain, serializers
import math
import numpy as np
import chainer.functions as F


class BaseModel(Chain):
    hyper_params = None
    params_prefix = None

    def __init__(self):
        super(BaseModel, self).__init__()
        self.config = {}

    
    @classmethod    
    def parse_args(cls, args):
        return { p: getattr(args, cls.params_prefix+'_'+p, None)
            for p in cls.hyper_params if getattr(args, cls.params_prefix+'_'+p, None) is not None }


    def save_model(self, fbase):
        with open(fbase+'.config', 'w') as f:
            for k, v in self.config.items():
                if type(v) is bool:
                    if v:
                        f.write('{}\n'.format(k))
                elif v is not None:
                    f.write('{}\n{}\n'.format(k, v))

        serializers.save_npz(fbase+'.npz', self)


    def base_onehot(self, base):
        if base in ['A' ,'a']:
            return np.array([1,0,0,0] , dtype=np.float32)
        elif base in ['U' ,'u']:
            return np.array([0,1,0,0] , dtype=np.float32)
        elif base in ['G' ,'g']:
            return np.array([0,0,1,0] , dtype=np.float32)
        elif base in ['C' ,'c']:
            return np.array([0,0,0,1] , dtype=np.float32)
        else:
            return np.array([0,0,0,0] , dtype=np.float32)


    def make_onehot_vector(self, seq, M=4):
        B = len(seq) # batch size
        # M = 4 # the number of bases
        N = max([len(s) for s in seq]) # the maximum length of sequences
        seq_vec = np.zeros((B, N, M), dtype=np.float32)
        for k, s in enumerate(seq):
            for i, base in enumerate(s):
                seq_vec[k, i, :] = self.base_onehot(base)
        return seq_vec


    def make_interval_vector(self, interval, bit_len, scale):
        i = int(math.log(interval, scale)+1.) if interval > 0 else 0
        return np.eye(bit_len, dtype=np.float32)[min(i, bit_len-1)]


    def make_input_vector(self, v, interval, bit_len=20, scale=1.5):
        B, N, _ = v.shape
        v_l = v[:, :-interval, :] # (B, N-interval, K)
        v_l = F.split_axis(v_l, 2, axis=2)[0]
        v_r = v[:, interval:, :]  # (B, N-interval, K)
        v_r = F.split_axis(v_r, 2, axis=2)[1]
        v_int = self.make_interval_vector(interval, bit_len, scale) # (bit_len,)
        v_int = self.xp.tile(v_int, (B, N-interval, 1)) # (B, N-interval, bit_len)
        x = F.concat((v_l+v_r, v_int), axis=2) # (B, N-interval, K/2+bit_len)
        return x


    def allowed_basepairs(self, seq, allowed_bp='canonical'):
        B = len(seq)
        N = max([len(s) for s in seq])

        if allowed_bp is None:
            return self.xp.ones((B, N, N), dtype=np.bool)

        elif isinstance(allowed_bp, str) and allowed_bp == 'canonical':
            allowed_bp = np.zeros((B, N, N), dtype=np.bool)
            canonicals = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}
            for k, s in enumerate(seq):
                s = s.upper()
                for j in range(len(s)):
                    for i in range(j):
                        if (s[i], s[j]) in canonicals and j-i>2:
                            allowed_bp[k, i, j] = allowed_bp[k, j, i] = True
            return allowed_bp

        elif isinstance(allowed_bp, list):
            a = np.zeros((B, N, N), dtype=np.bool)
            for k, bp in enumerate(allowed_bp):
                for i, j in bp:
                    a[k, i, j] = a[k, j, i] = True
            return a
