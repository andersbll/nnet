import numpy as np
from .helpers import sigmoid, sigmoid_d, relu, relu_d, tanh, tanh_d


class Layer(object):
    def _setup(self, input_shape, rng):
        """ Setup layer with parameters that are unknown at __init__(). """
        pass

    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, output_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def output_shape(self, input_shape):
        """ Calculate shape of this layer's output.
        input_shape[0] is the number of samples in the input.
        input_shape[1:] is the shape of the feature.
        """
        raise NotImplementedError()


class LossMixin(object):
    def loss(self, output, output_pred):
        """ Calculate mean loss given output and predicted output. """
        raise NotImplementedError()

    def input_grad(self, output, output_pred):
        """ Calculate input gradient given output and predicted output. """
        raise NotImplementedError()


class ParamMixin(object):
    def params(self):
        """ Layer parameters. """
        raise NotImplementedError()

    def param_grads(self):
        """ Get layer parameter gradients as calculated from bprop(). """
        raise NotImplementedError()

    def param_incs(self):
        """ Get layer parameter steps as calculated from bprop(). """
        raise NotImplementedError()


class Linear(Layer, ParamMixin):
    def __init__(self, n_out, weight_scale, weight_decay=0.0):
        self.n_out = n_out
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay

    def _setup(self, input_shape, rng):
        n_input = input_shape[1]
        W_shape = (n_input, self.n_out)
        self.W = rng.normal(size=W_shape, scale=self.weight_scale)
        self.b = np.zeros(self.n_out)

    def fprop(self, input):
        self.last_input = input
        return np.dot(input, self.W) + self.b

    def bprop(self, output_grad):
        n = output_grad.shape[0]
        self.dW = np.dot(self.last_input.T, output_grad)/n - self.weight_decay*self.W
        self.db = np.mean(output_grad, axis=0)
        return np.dot(output_grad, self.W.T)

    def params(self):
        return self.W, self.b

    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        # undo weight decay to get gradient
        gW = self.dW+self.weight_decay*self.W
        return gW, self.db

    def output_shape(self, input_shape):
        return (input_shape[0], self.n_out)


class Activation(Layer):
    def __init__(self, type):
        if type == 'sigmoid':
            self.fun = sigmoid
            self.fun_d = sigmoid_d
        elif type == 'relu':
            self.fun = relu
            self.fun_d = relu_d
        elif type == 'tanh':
            self.fun = tanh
            self.fun_d = tanh_d
        else:
            raise ValueError('Invalid activation function.')

    def fprop(self, input):
        self.last_input = input
        return self.fun(input)

    def bprop(self, output_grad):
        return output_grad*self.fun_d(self.last_input)

    def output_shape(self, input_shape):
        return input_shape


class LogRegression(Layer, LossMixin):
    """ Softmax layer with cross-entropy loss function. """
    def fprop(self, input):
        e = np.exp(input - np.amax(input, axis=1, keepdims=True))
        return e/np.sum(e, axis=1, keepdims=True)

    def bprop(self, output_grad):
        raise NotImplementedError(
            'LogRegression does not support back-propagation of gradients. '
            + 'It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, Y, Y_pred):
        # Assumes one-hot encoding.
        return -(Y - Y_pred)

    def loss(self, Y, Y_pred):
        # Assumes one-hot encoding.
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        Y_pred /= Y_pred.sum(axis=1, keepdims=True)
        loss = -np.sum(Y * np.log(Y_pred))
        return loss / Y.shape[0]

    def output_shape(self, input_shape):
        return input_shape
