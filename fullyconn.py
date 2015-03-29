import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class InputLayer():
    def __init__(self, name, dimension, input=None):
        self.name   = name
        if input is None:
            input = T.matrix("input")
        self.input = input
        self.params = []
        self.dimension = dimension

    def output(self):
        return self.input

    def regularization(self):
        return None

    def initialize(self, rng):
        pass

class FullyConnLayer():
    def __init__(self,
                 name,
                 dimension,
                 input=None,
                 activation=T.nnet.sigmoid):
        self.activation = activation
        self.dimension = dimension
        self.input = input
        self.name = name
        self.W = None
        self.b = None
        self.params = []

    def regularization(self):
        assert (self.W is not None), "The layer named '{0}' is not initialized"
        return T.sum(self.W ** 2)

    def initialize(self, rng = None, W = None, b = None):
        if W is None:
            assert (rng is not None), "Please provide a RNG if you do not set the weights manually"
            initial_W = np.asarray(
                rng.uniform(
                    low  = -4 * np.sqrt(6. / (self.input.dimension + self.dimension)),
                    high = 4 * np.sqrt(6. / (self.input.dimension + self.dimension)),
                    size = (self.input.dimension, self.dimension)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name="W_" + self.name, borrow=True)

        if b is None:
            assert (rng is not None), "Please provide a RNG if you do not set the bias manually"
            initial_b = np.zeros(
                (self.dimension,),
                dtype=theano.config.floatX
            )
            b = theano.shared(value=initial_b, name="b_" + self.name, borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

    def connect(self, input):
        self.input = input

    def output(self):
        assert (self.input is not None), "The layer named '{0}' is not connected".format(self.name)
        assert (self.W is not None and self.b is not None), "The layer named '{0}' is not initialized".format(self.name)

        return self.activation(T.dot(self.input.output(), self.W) + self.b)

class MLP():
    def __init__(self):
        self.layers = []
        self.params = []
        self.rng = np.random.RandomState(42)

    def add_layer(self, layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])
        self.layers[-1].initialize(self.rng)
        self.params = self.params + self.layers[-1].params

    def output(self):
        assert (len(self.layers) > 0), "The network needs to contain at least one layer."
        return self.layers[-1].output()

    def remove_last_layer(self):
        if len(self.layers) > 0:
            self.layers.pop()
            self.params = []
            for layer in self.layers:
                self.params += layer.params

    def build_train(self, learning_rate, regularization_factor, target_weights=None):
        assert (len(self.layers) > 0), "The network needs to contain at least one layer."
        labels = T.matrix("labels", dtype=theano.config.floatX)

        if target_weights is None:
            cost = T.sum((self.output() - labels) ** 2)
        else:
            cost = T.sum((self.output() - labels) ** 2, axis=0)
            cost = T.sum(cost * target_weights)

        for layer in self.layers:
            if layer.regularization() is not None:
                cost = cost + regularization_factor * layer.regularization()
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [
                (param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)
                ]

        return theano.function(
            inputs = [self.layers[0].input, labels],
            outputs=cost,
            updates=updates
        )

    def build_eval(self):
        assert (len(self.layers) > 0), "The network needs to contain at least one layer."
        return theano.function(
            inputs = [self.layers[0].input],
            outputs = self.output()
        )
