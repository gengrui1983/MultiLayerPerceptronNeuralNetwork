import numpy as np

class Activation(object):

    def __init__(self, activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_derivative
        elif activation == 'l_relu':
            self.f = self.__leaky_relu
            self.f_deriv= self.__leaky_relu_derivative
        elif activation == None:
            self.f = None
            self.f_deriv = None

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        # a = logistic(x)
        return a * (1 - a)

    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_derivative(self, a):
        dx = np.ones_like(a)
        dx[a <= 0] = 0
        return dx

    def __leaky_relu(self, x, alpha=0.01):
        return np.maximum(x * alpha, x)

    def __leaky_relu_derivative(self, a, alpha=0.01):
        dx = np.ones_like(a)
        dx[a < 0] = alpha
        return dx
