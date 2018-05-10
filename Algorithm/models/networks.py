import numpy as np

from models.layer import HiddenLayer
import time


class MLP:

    def __init__(self, layers, dropouts, activation='tanh', norm=None, update_type=None):
        """
        MLP is a class to create a multiple layers neural network.

        :type layers: [int]
        :param layers: A list containing the number of units in each layer.
        Should be at least two values

        :type dropouts: [int]
        :param dropouts: The activation function to be used. Can be

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer, defaults to relu

        :type norm: string
        :param norm: normalisation module to be applied
        if norm = 'bn', MLP is run with batch normalisation;
        if norm = 'wn', MLP is run with weight normalisation;
        Otherwise run without batch normalisation and weight normalisation

        :type update_type: string
        :param update_type: the method used to update the parameters
        if update_type = "momentum", use momentum update
        if update_type = "nes_momentum", Nesterov Momentum
        if update_type = None, just update with a small fraction of the derivatives
        """

        if norm == "bn" and update_type == "nes_momentum":
            raise Exception("The Batch Normalization with Nesterov Momentum is not supported for this application.")

        ### initialize layers
        self.layers = []
        self.params = []
        self.epsilon = 1e-10
        self.dropout_masks = []
        self.dropouts = []
        # activation function
        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation=activation,
                                           norm=norm, update_type=update_type,
                                           dropout=dropouts[i],
                                           is_last=(i == len(layers) - 2)))

    def forward(self, input, test_mode=False):
        '''
        A function to run forward procedure for all layers

        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)

        '''
        # for each layer
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # run a forward process
            output = layer.forward(input)
            if not test_mode:
                # perform dropout
                if i != len(self.layers) - 1 and layer.dropout_p != -1:
                    mask = layer.dropout(output)
                    output *= mask
                    self.dropout_masks.append(mask)
            input = output
        return output

    def backward(self, delta):
        """
        A function to run backward procedure for all layers

        :type delta: numpy.array
        :param delta: the derivatives return back to the layers

        """
        # for each layer (reverse the order)
        for i in reversed(range(len(self.layers))):
            delta = self.layers[i].backward(
                delta,
                self.layers[i - 1] if i > 0 else None
            )

            if i != 0 and self.layers[i].dropout_p != -1:
                delta *= self.dropout_masks[i - 1]

    def __softmax(self, x):
        """
        softmax function

        :type x: np.array
        :param x: input

        :return: the softmax transformation of the input
        """
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / (np.sum(exps, axis=1, keepdims=True))

    def cross_entropy(self, y, y_hat):
        #transform the predicted y with softmax function
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
        """
        A function to update the parameters for all layers

        :type my: float
        :param my: an additional hyperparameter for momentum update

        :type lr: float
        :param lr: the learning rate of the model, is the step size of a parameter update

        """
        for layer in self.layers:
            layer.update(my, lr)

    def iterate_minibatches(self, inputs, y, batchsize, shuffle=False):
        """
        A function to create all the minibatches

        :type inputs: np.arrays
        :param inputs: the data input

        :type y: np.arrays
        :param y: the actual classes

        :type batchsize: int
        :param batchsize: the size of the batch

        :type shuffle: bool
        :param shuffle: whether to shuffle the data or not

        :return: a list of each mini-batch, each batch is a list of the input data and the actuall classes
        """
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
        A function to run the training process.

        :type X: np.array
        :param X: Input data or features for training

        :type y: np.array
        :param y: Input targets for training

        :type data_val: np.array
        :param data_val: Input data or features for validation

        :type y_val: np.array
        :param y_val: Input targets for validation

        :type my: float
        :param my: an additional hyperparameter for momentum update

        :type learning_rate: float
        :param learning_rate: the learning rate of the model, is the step size of a parameter update

        :type batchsize: int
        :param batchsize: the size of the batch

        :type epochs: int
        :param epochs: number of times the dataset is presented to the network for learning

        :return: a list of training accuracy of all epochs,
        a list of training loss of all epochs,
        a list of testing accuracy of all epochs,
        a list of testing loss of all epochs
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
            # for each mini batch
            for batch in self.iterate_minibatches(X, y, batchsize=batchsize, shuffle=True):
                # get the training input data and the target y
                X_batch, y_batch = batch
                # run forward procedure
                y_hat = self.forward(X_batch)
                # calculate loss and the derivatives to pass to the layers
                loss, delta, reg_loss = self.cross_entropy(y_batch, y_hat)
                # run backward procedure
                self.backward(delta)
                # update parameters
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
        """
        A function to make prediction.
        :type X: np.array
        :param X: Input data or features for prediction

        :type y: np.array
        :param y: Input targets to be compared with the prediction

        :return: the accuracy and loss of the prediction
        """
        score = self.forward(input)
        pred = np.argmax(score, axis=1)
        return score, pred

    def evaluate(self, pred, y):
        score, pred = self.predict(pred)
        loss, delta, _ = self.cross_entropy(y, score)
        acc = np.mean(pred == y)
        print("Testing\t\tloss:\t{:0.10f}\t\tacc:\t{:0.2f}%".format(loss, acc * 100))
        return loss, acc
