from models.activation import Activation
import numpy as np
from numpy import linalg as LA
from .weight_init import WeightInit


class HiddenLayer(object):
    def __init__(self, n_in, n_out, is_last=False,
                 activation='tanh', norm=None, dropout=0.5, update_type=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.n_in = n_in
        self.n_out = n_out
        self.epsl = 1e-5
        self.input, self.output = None, None

        self.activation = Activation(activation).f
        self.activation_deriv = Activation(activation).f_deriv
        self.activation_name = activation

        self.norm = norm
        self.update_type = update_type

        # The cached linear result of net(x) = W.X + b
        self.cache_li_output = None

        # Params for weight normalization
        self.is_init = True
        self.grad_V, self.V = None, None
        self.g_wn, self.b_wn = None, None

        # Params for batch normalization
        self.beta, self.gamma = None, None
        self.grad_beta, self.grad_gamma = None, None
        self.mean, self.var, self.x, self.x_norm = None, None, None, None

        # Initialise the weights and bias
        self.W = WeightInit("He", n_in, n_out).get_weights()
        self.b = np.zeros((1, n_out))

        # Momentum
        self.velocity = np.zeros((n_in, n_out))

        # Gradient of Weights and bias
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        # If the layer is the last layer
        self.is_last = is_last

        # The layer mask and the prob for dropping out.
        self.layer_mask = None
        self.dropout_p = dropout

    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''

        if self.norm == "wn":
            if self.is_init:
                # Rescale the weights and bias by calculating the mean and std of the linear net output.
                init_scale = 1
                self.V = np.random.normal(loc=0, scale=0.05, size=(input.shape[1], self.n_out))
                v_norm = LA.norm(self.V, axis=0)
                x_init = np.dot(input, self.V) / v_norm
                m_init = np.mean(x_init, axis=0)
                v_init = np.std(x_init, axis=0)
                scale_init = init_scale / np.sqrt(v_init + 1e-10)
                self.g_wn = scale_init
                self.b_wn = -m_init * scale_init
                x_init = np.reshape(scale_init, [1, self.n_out]) * (x_init - np.reshape(m_init, [1, self.n_out]))
                lin_output = x_init
                self.is_init = False
            else:
                t = np.dot(input, self.V)
                scalar = self.g_wn / LA.norm(self.V, axis=0)
                lin_output = np.reshape(scalar, [1, self.n_out]) * t + np.reshape(self.b_wn, [1, self.n_out])

        elif self.norm == "bn":
            if self.beta is None:
                self.beta = np.zeros((self.n_out,))

            if self.gamma is None:
                self.gamma = np.ones((self.n_out,))

            # calculate intermediate value
            self.x = np.dot(input, self.W)
            self.mean = np.mean(self.x, axis=0)
            self.var = np.var(self.x, axis=0)
            self.x_norm = (self.x - self.mean) / np.sqrt(self.var + 1e-8)

            lin_output = self.gamma * self.x_norm + self.beta

        else:
            # Linear output
            lin_output = np.dot(input, self.W) + self.b

        self.cache_li_output = lin_output

        self.output = (
            lin_output if self.activation is None or self.is_last
            else self.activation(lin_output)
        )

        self.input = input
        return self.output

    def _get_weight(self):
        return self.g_wn / LA.norm(self.V, axis=0) * self.V

    def backward(self, delta, prev_layer=None):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0, keepdims=True)

        if self.norm == "wn":
            self.W = self._get_weight()
            v_norm = LA.norm(self.V, axis=0)
            m_w = self.W.dot(self.W.T) / LA.norm(self.W) ** 2
            self.grad_V = self.g_wn / v_norm * (np.identity(m_w.shape[0]) - m_w).dot(self.grad_W)

            # return delta_ for next layer
            cache_output = self.input if prev_layer is None else prev_layer.cache_li_output
            delta_ = delta.dot(self.W.T) * self.activation_deriv(cache_output)
        elif self.norm == "bn":
            # calculate the gradient of gamma and beta
            N, D = self.x.shape
            x_mean = self.x - self.mean
            std_inv = 1. / np.sqrt(self.var + 1e-8)

            dx_norm = delta * self.gamma
            dvar = np.sum(dx_norm * x_mean, axis=0) * -.5 * std_inv ** 3
            dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2. * x_mean, axis=0)

            dxw = (dx_norm * std_inv) + (dvar * 2 * dmean / N) + (dmean / N)

            self.grad_gamma = np.sum(delta * self.x_norm, axis=0)
            self.grad_beta = np.sum(delta, axis=0)
            self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(dxw))
            # calculate delta to pass
            delta_ = dxw.dot(self.W.T)

        else:
            # return delta_ for next layer
            cache_output = self.input if prev_layer is None else prev_layer.cache_li_output
            delta_ = delta.dot(self.W.T) * self.activation_deriv(cache_output)

        return delta_

    def dropout(self, input, rng=None):
        if rng is None:
            rng = np.random.RandomState(None)

        mask = rng.binomial(size=input.shape, n=1, p=1 - self.dropout_p)
        return mask

    def update(self, my, lr):
        if self.update_type == "nes_momentum":
            prev_v = self.velocity

            if self.norm == "wn":
                # Nesterov Momentum + Weight Normalization
                self.velocity = my * self.velocity + lr * self.grad_V
                self.V -= -my * prev_v + (1 + my) * self.velocity
            else:
                # Nesterov Momentum without Normalization
                self.velocity = my * self.velocity + lr * self.grad_W
                self.W -= -my * prev_v + (1 + my) * self.velocity

        elif self.update_type == "momentum":
            if self.norm == "wn":
                # Momentum + Weight Normalization
                self.velocity = my * self.velocity + lr * self.grad_V
                self.V -= self.velocity

            elif self.norm == "bn":
                # Momentum + Batch Normalization
                self.velocity = my * self.velocity + lr * self.grad_W
                self.W -= self.velocity
                self.gamma += -lr * self.grad_gamma
                self.beta += -lr * self.grad_beta

            else:
                # Momentum
                self.velocity = my * self.velocity + lr * self.grad_W
                self.W -= self.velocity
                self.b -= lr * self.grad_b

        else:
            if self.norm == "wn":
                # Weight Normalization
                self.V -= lr * self.grad_V
            elif self.norm == "bn":
                self.gamma += -lr * self.grad_gamma
                self.beta += -lr * self.grad_beta
                pass
            else:
                # Normal updates, without normalization and momentum
                self.W -= lr * self.grad_W
                self.b -= lr * self.grad_b
