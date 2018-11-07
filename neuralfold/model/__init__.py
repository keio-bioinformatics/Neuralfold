import pickle
from chainer import serializers
from .mlp import MLP

def load_model(f):
    with open(f+'.pickle', 'rb') as fp:
        klass_name = pickle.load(fp)
        hyper_params = pickle.load(fp)
        try:
            klass = globals()[klass_name]
            model = klass(**hyper_params)
        except KeyError:
            raise RuntimeError("{} is unknown model class.".format(klass_name))

    serializers.load_npz(f+'.npz', model)
    return model
