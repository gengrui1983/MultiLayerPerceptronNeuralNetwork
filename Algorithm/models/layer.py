from models.activation import Activation
import numpy as np
from numpy import linalg as LA


class HiddenLayer(object):
    def __init__(self, n_in, n_out, W=None, b=None, is_last=False,
                 activation='tanh', dropout=0.5):
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
        self.input = None
        self.activation = Activation(activation).f
        self.activation_deriv = Activation(activation).f_deriv
        self.activation_name = activation
        self.dropout_p = dropout
        self.cache_li_output = None
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        # self.W = np.random.uniform(
        #     low=-np.sqrt(6. / (n_in + n_out)),
        #     high=np.sqrt(6. / (n_in + n_out)),
        #     size=(n_in, n_out)
        # )

        # Xavier Initialization
        # self.W = np.random.normal(
        #     mean
            # loc=0,
            # scale=np.math.sqrt(2.0 / n_in),
            # size=(n_in, n_out)
        # )

        # He Initialization
        self.W = np.random.randn(n_in, n_out) / np.sqrt(n_in / 2)

        # For weight norm
        self.is_init = True
        self.g_wn, self.b_wn = None, None

        # Kaiming Initialization
        # gain = 1
        # std = gain * np.sqrt(1.0 / n_in)
        # a = np.sqrt(3.0) * std
        # self.W = np.random.uniform(-a, a)

        self.b = np.zeros((1, n_out))
        self.velocity = np.zeros((n_in, n_out))

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        self.is_last = is_last
        self.layer_mask = None

    def forward(self, input):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''

        if self.is_init:
            # Rescale the weights and bias by calculating the mean and std of the linear net output.
            init_scale = 1
            self.V = np.random.normal(loc=0, scale=0.05, size=(input.shape[1], self.n_out))
            V_norm = LA.norm(self.V, axis=0)
            x_init = np.dot(input, self.V) / V_norm
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
            scaler = self.g_wn / LA.norm(self.V, axis=0)
            lin_output = np.reshape(scaler, [1, self.n_out]) * t + np.reshape(self.b_wn, [1, self.n_out])

        self.cache_li_output = lin_output

        # Linear output
        # lin_output = np.dot(input, self.W) + self.b

        self.output = (
            lin_output if self.activation is None or self.is_last
            else self.activation(lin_output)
        )

        self.input = input
        return self.output

    def get_weight(self):
        return self.g_wn / LA.norm(self.V, axis=0) * self.V

    def backward(self, delta, prev_layer=None):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0, keepdims=True)

        self.W = self.get_weight()

        V_norm = LA.norm(self.V, axis=0)
        m_w = self.W.dot(self.W.T) / LA.norm(self.W) ** 2
        self.grad_V = self.g_wn / V_norm * (np.identity(m_w.shape[0]) - m_w).dot(self.grad_W)

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
        # Nesterov Momentum
        # prev_v = self.velocity
        # self.velocity = my * self.velocity + lr * self.grad_W
        # self.W -= -my * prev_v + (1 + my) * self.velocity

        # Nesterov Momentum + Weight Norm
        prev_v = self.velocity
        self.velocity = my * self.velocity + lr * self.grad_V
        self.V -= -my * prev_v + (1 + my) * self.velocity

        # Weight Norm
        # self.V -= lr * self.grad_V

        # Momentum + Weight Norm
        # self.velocity = my * self.velocity + lr * self.grad_V
        # self.V -= self.velocity
        # self.b_wn -= lr * self.grad_b

        # Momentum
        # self.velocity = my * self.velocity + lr * self.grad_W
        # self.W -= self.velocity
        # self.b -= lr * self.grad_b
