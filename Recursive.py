import chainer.links as L
import chainer.functions as F
import chainer

class Recursive_net(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        # パラメータを持つ層の登録
        super(Recursive_net, self).__init__(
            l1 = L.Linear(None , n_out),
            l2 = L.Linear(None , 1)
        )

    def __call__(self, x , inner = False):
        # データを受け取った際のforward計算を書く
        if inner:
            h = F.relu(self.l1(x))
        else:
            h = F.relu(self.l2(x))
        return h

gpu_id = 0

model = MLP()
model.to_gpu(gpu_id)
