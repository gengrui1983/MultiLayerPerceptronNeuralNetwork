import numpy as np

from models.layer import HiddenLayer
import time


class MLP:
    """
    """

    def __init__(self, layers, dropouts, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        ### initialize layers
        self.layers = []
        self.params = []
        self.epsilon = 1e-10
        self.dropout_masks = []
        self.dropouts = []

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation=activation,
                                           dropout=dropouts[i],
                                           is_last=(i == len(layers) - 2)))

    def forward(self, input, test_mode=False):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            output = layer.forward(input)

            if not test_mode:
                if i != len(self.layers) - 1 and layer.dropout_p != -1:
                    mask = layer.dropout(output)
                    output *= mask
                    self.dropout_masks.append(mask)

            input = output
        return output

    def backward(self, delta):
        for i in reversed(range(len(self.layers))):
            delta = self.layers[i].backward(
                delta,
                self.layers[i-1] if i > 0 else None
            )

            if i != 0 and self.layers[i].dropout_p != -1:
                delta *= self.dropout_masks[i-1]

    def __softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / (np.sum(exps, axis=1, keepdims=True))

    def cross_entropy(self, y, y_hat):

        probs = self.__softmax(y_hat)

        m = y_hat.shape[0]
        log_likelihood = -np.log(probs[range(m), y])
        loss = np.sum(log_likelihood) / m

        reg_loss = 0
        reg = 1e-4
        for l in self.layers:
            reg_loss += np.sum(l.W ** 2) * reg
        loss += reg_loss / len(self.layers)

        # Calculate the sensitivity of the output units.
        delta_ = probs
        delta_[range(m), y] -= 1
        delta_ /= m

        return loss, delta_, reg_loss

    def update(self, my, lr):
        for layer in self.layers:
            layer.update(my, lr)

    def iterate_minibatches(self, inputs, y, batchsize, shuffle=False):
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt, :], y[excerpt]

    def fit(self, X, y, data_val, y_val, learning_rate=0.1, my=0.9, epochs=100, batchsize=1000):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = np.array(X)
        y = np.array(y)

        to_return = np.zeros(epochs)

        prev_time = time.time()
        for k in range(epochs):
            itr = 0
            for batch in self.iterate_minibatches(X, y, batchsize=batchsize, shuffle=True):

                X_batch, y_batch = batch

                y_hat = self.forward(X_batch)

                loss, delta, reg_loss = self.cross_entropy(y_batch, y_hat)

                self.backward(delta)
                self.update(my, learning_rate)

                to_return[k] = np.mean(loss)

                if itr % 100 == 0:
                    pred = np.argmax(y_hat, axis=1)
                    print("{}. loss: {}, reg_loss: {}, accuracy:{}".format(k, to_return[k], reg_loss, np.mean(pred == y_batch)))
                    self.predict(data_val, y_val)

                itr += 1

        return to_return

    def predict(self, input, y):
        score = self.forward(input, test_mode=True)
        pred = np.argmax(score, axis=1)
        print("testing accuracy: {}".format(np.mean(pred == y)))
