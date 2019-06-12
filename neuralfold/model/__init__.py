import pickle
from chainer import serializers
from .cnn import CNN
from .mlp import MLP
from .rnn import RNN
from .wncnn import WNCNN
from .wncnn2d import WNCNN2D

def load_model(f, args):
    try:
        klass = globals()[args.learning_model]
        model = klass(**klass.parse_args(args))
    except KeyError:
        raise RuntimeError("{} is unknown model class.".format(args.learning_model))

    serializers.load_npz(f, model)
    return model
