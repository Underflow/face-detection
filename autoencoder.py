#! env python2.7
import numpy as np
import theano
from theano import tensor
from theano import function
from theano import pp
from theano import grad
from theano.tensor import nnet
from theano.tensor import dot
import sys
import math

class Layer:
    x = tensor.vector('input')
    lout = tensor.vector('last_output')
    w = tensor.matrix('weights')
    w2 = tensor.matrix('weights')
    err = tensor.vector('err')

    activation = function([x, w], tensor.nnet.sigmoid(dot(w, x)))
    sigp = tensor.exp(-lout) / tensor.sqr((1 + tensor.exp(-lout)))
    fun_err = function([lout, w2, err], tensor.dot(w2.T, err) * sigp)
    delta_w = function([err, lout], err.dimshuffle((0, 'x')) * lout)

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) / 10
        self.learning_rate = 1
        self.last_output = []
        self.error_val = 0

    def error(self, n_error, n_weights):
        self.error_val = Layer.fun_err(self.last_output, n_weights, n_error)
        return self.error_val

    def eval(self, input_val):
        if np.shape(self.weights)[1] != np.shape(input_val)[0]:
            raise ValueError("Wrong input size.")
        self.last_output = Layer.activation(input_val, self.weights)
        return self.last_output

class Neuralnet(object):
    y = tensor.vector("output")
    t = tensor.vector("exected")
    ll_error = function([t, y], t - y)

    def __init__(self):
        self.output_dim = None
        self.layers = []

    def add_layer(self, dim):
        if dim <= 0:
            raise ValueError("A layer cannot be empty.")
        input_dim = self.output_dim if self.output_dim else dim
        self.output_dim = dim
        self.layers.append(Layer(input_dim, dim))

    def eval(self, X):
        output_vector = X
        for layer in self.layers:
            output_vector = layer.eval(output_vector)
        return output_vector

    def update_progress(self, progress, error, valid_error):
        sys.stdout.write('\rBatch : [{0}] {1}% - err: {2}%, val-err: {3}%'.format('#'*(progress/10) + ' '*(10 - progress/10), progress, error, valid_error))

        sys.stdout.flush()

    def train(self, examples, expected, steps, rate):
        if len(examples) != len(expected):
            raise ValueError("There is not the same number of examples and labels")
        if len(self.layers) <= 0:
            raise Exception("Impossible to train an empty neural network")
        print("Training neural network. {0} learning steps, learning rate : {1}, {2} examples".format(steps, rate, len(examples)))

        training_examples = len(examples) / 2
        for i in range(0, steps):
            error = 0
            valid_error = 0
            for example, label in zip(examples[training_examples:], expected[:training_examples]):
                out = self.eval(example)
                last_error = Neuralnet.ll_error(label, out)
                valid_error += np.sum(last_error ** 2)

            # On-line training with stochastic gradient descent
            for example, label in zip(examples[0:training_examples], expected[0:training_examples]):
                # Propagate signal
                out = self.eval(example)
                last_error = Neuralnet.ll_error(label, out)
                error += np.sum(last_error ** 2)
                self.layers[-1].error_val = last_error
                last_weights = self.layers[-1].weights

                # Back-propagation
                for layer in reversed(self.layers[:-1]):
                    last_error = layer.error(last_error, last_weights)
                    last_weights = layer.weights

                # Weight update
                last_output = example
                for layer in self.layers:
                    layer.weights += rate * Layer.delta_w(layer.error_val, last_output)
                    last_output = layer.last_output
            error = round(math.sqrt(error / training_examples / 400) * 100, 4)
            valid_error = round(math.sqrt(valid_error / training_examples / 400) * 100, 4)
            self.update_progress(int(float(i + 1) / steps * 100), error, valid_error)
        print("")


class Autoencoder(Neuralnet):
    def train(self, dataset, reduction, steps, rate):
        print("Setting-up an autoencoder. Dimensionality reduction : {0}".format(reduction))
        self.layers = []
        self.add_layer(np.shape(dataset)[1])
        self.add_layer(np.shape(dataset)[1] - reduction)
        self.add_layer(np.shape(dataset)[1])
        # Learn the identity function to proceed a dimensionality reduction
        super(Autoencoder, self).train(dataset, dataset, steps, rate)
         # Because a trained autoencoder is not the identity function
        #self.layers.pop()
