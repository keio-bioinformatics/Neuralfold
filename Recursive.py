import chainer.links as L
import chainer.functions as F
import chainer

class Recursive_net(chainer.Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        # パラメータを持つ層の登録
        super(Recursive_net, self).__init__(
            l1=L.Linear(None, n_mid_units),
            l2=L.Linear(n_mid_units, n_mid_units),
            l3=L.Linear(n_mid_units, n_out),
        )

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

gpu_id = 0

model = MLP()
model.to_gpu(gpu_id)
