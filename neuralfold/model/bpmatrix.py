import numpy as np

class BatchedBPMatrix:
    def __init__(self, diagonals):
        self.diagonals = diagonals

    def __getitem__(self, k):
        return BPMatrix(self.diagonals, k)

    @property
    def array(self):
        xp = self.xp
        N = len(self.diagonals)
        B = self.diagonals[1].shape[0]
        dtype = self.diagonals[1].dtype
        bpp = xp.zeros((B, N, N), dtype=dtype)
        for k in range(1, N):
            x = self.diagonals[k].array
            x = xp.hstack((xp.zeros((B, k), dtype=dtype), x))
            x = xp.tile(x, N).reshape(B, N, N)
            cond = xp.diag(xp.ones(N-k, dtype=np.bool), k=k)
            bpp = xp.where(cond, x, bpp)
        return bpp

    @property
    def xp(self):
        return self.diagonals[1].xp


class BPMatrix:
    def __init__(self, diagonals, k):
        self.diagonals = diagonals
        self.k = k

    def __getitem__(self, ij):
        i, j = ij
        return self.diagonals[j-i][self.k, i]

    @property
    def xp(self):
        return self.diagonals[1].xp
