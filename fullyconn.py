import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class FullyConnLayer():
    def __init__(self,
                 rng,
                 name,
                 dimension,
                 input_dimension,
                 theano_rng = None,
                 input=None,
                 W=None,
                 b=None,
                 activation=T.nnet.sigmoid):

        self.activation = activation
        self.dimension = dimension
        self.input_dimension = input_dimension
        if input is None:
            self.input = T.dmatrix(name="input")
        else:
            self.input = input

        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2 ** 30))

        if W is None:
            initial_W = np.asarray(
                rng.uniform(
                    low  = -4 * np.sqrt(6. / (input_dimension + dimension)),
                    high = 4 * np.sqrt(6. / (input_dimension + dimension)),
                    size = (input_dimension, dimension)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name="W_" + name, borrow=True)

        if b is None:
            initial_b = np.zeros(
                (dimension,),
                dtype=theano.config.floatX
            )
            b = theano.shared(value=initial_b, name="b_" + name, borrow=True)

        self.theano_rng = theano_rng
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.output = activation(T.dot(self.input, self.W) + self.b)

class MLP():
    def __init__(self, input_dimension, input=None):
        self.rng = np.random.RandomState(42)
        if input is None:
            self.input = T.matrix("input")
        else:
            self.input = input
        self.output = self.input
        self.input_dimension = input_dimension
        self.layers = []
        self.params = []

    def add_layer(self, name, dimension):
        if self.layers == []:
            self.layers.append(FullyConnLayer(self.rng,
                                              name,
                                              dimension,
                                              self.input_dimension,
                                              input=self.input))
        else:
            self.layers.append(FullyConnLayer(self.rng,
                                              name,
                                              dimension,
                                              self.layers[-1].dimension,
                                              input=self.layers[-1].output))
        self.output = self.layers[-1].output
        self.params = self.params + self.layers[-1].params
        
    def remove_last_layer(self):
        if len(self.layers) == 1:
            self.params = []
            self.output = self.input
        else:
            self.layers.pop()
            self.params = []
            for layer in self.layers:
                self.params += layer.params
            self.output = self.layers[-1].output

    def build_train(self, learning_rate, regularization):
        labels = T.matrix("labels", dtype=theano.config.floatX)
        cost = T.sum((self.output - labels) ** 2)
        for layer in self.layers:
            cost = cost + regularization * T.sum(layer.W ** 2)
        gparams = [T.grad(cost, param) for param in self.params]
        updates = [
                (param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)
                ]

        return theano.function(
            inputs = [self.input, labels],
            outputs=cost,
            updates=updates
        )

    def build_eval(self):
        return theano.function(
            inputs = [self.input],
            outputs = self.output
        )
