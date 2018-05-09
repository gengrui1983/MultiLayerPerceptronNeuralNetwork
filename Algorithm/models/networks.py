import numpy as np

from models.layer import HiddenLayer
import time


class MLP:
    """
    """

    def __init__(self, layers, dropouts, activation='tanh', norm=None, update_type=None):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """

        if norm == "bn" and update_type == "nes_momentum":
            raise Exception("The Batch Normalization with Nesterov Momentum is not supported for this application.")

        ### initialize layers
        self.layers = []
        self.params = []
        self.epsilon = 1e-10
        self.dropout_masks = []
        self.dropouts = []

        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation=activation,
                                           norm=norm, update_type=update_type,
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
                self.layers[i - 1] if i > 0 else None
            )

            if i != 0 and self.layers[i].dropout_p != -1:
                delta *= self.dropout_masks[i - 1]

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

        # the return loss result
        train_loss_return = []
        train_acc_return = []
        # the return acc result
        test_loss_return = []
        test_acc_return = []

        to_return = np.zeros(epochs)

        prev_time = time.time()
        for k in range(epochs):
            itr = 0
            batch_num = X.shape[0]

            for batch in self.iterate_minibatches(X, y, batchsize=batchsize, shuffle=True):

                X_batch, y_batch = batch

                y_hat = self.forward(X_batch)

                loss, delta, reg_loss = self.cross_entropy(y_batch, y_hat)

                self.backward(delta)
                self.update(my, learning_rate)

                # at the end of each epoch
                if itr % batch_num == 0:
                    # get the predict result
                    pred = np.argmax(y_hat, axis=1)
                    acc = np.mean(pred == y_batch)
                    # record training result
                    train_acc_return.append(acc)
                    train_loss_return.append(loss)
                    # print the result of the last batch at iteration
                    print("Epoch: {}\nTraining\tloss:\t{:0.10f}\t\tacc:\t{:0.2f}%".format(k + 1,
                                                                                         loss,
                                                                                         np.mean(pred == y_batch) * 100
                                                                                         ))
                    # print the result of the testing
                    tloss, tacc = self.evaluate(data_val, y_val)
                    # record testing result
                    test_loss_return.append(tloss)
                    test_acc_return.append(tacc)

                itr += 1
                self.dropout_masks.clear()

        return train_acc_return, train_loss_return, test_acc_return, test_loss_return

    def predict(self, input):
        score = self.forward(input)
        pred = np.argmax(score, axis=1)
        return score, pred

    def evaluate(self, pred, y):
        score, pred = self.predict(pred)
        loss, delta, _ = self.cross_entropy(y, score)
        acc = np.mean(pred == y)
        print("Testing\t\tloss:\t{:0.10f}\t\tacc:\t{:0.2f}%".format(loss, acc * 100))
        return loss, acc