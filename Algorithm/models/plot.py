import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, epoch, train_acc, train_loss, test_acc, test_loss, prefix=None):
        self.epoch = epoch
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.test_acc = test_acc
        self.test_loss = test_loss
        self.name_prefix = prefix

    def plot(self):
        x_index_test = np.arange(1.0, self.epoch + 1)
        # x_index_train=np.arange(0.0, epoch, 1.0 / num_of_batch)
        print(len(self.test_loss), len(x_index_test))
        # ploting the loss
        prefix = "_{}".format(self.name_prefix) if self.name_prefix is not None else ""

        plt.figure(1)
        plt.plot(x_index_test, self.train_loss)
        plt.plot(x_index_test, self.test_loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title('Loss of Training and Testing (Batch Normalisation)')
        plt.legend(['training', 'testing'], loc='upper left')
        plt.savefig('loss{}.png'.format(prefix))

        # ploting the training
        plt.figure(2)
        plt.plot(x_index_test, self.train_acc)
        plt.plot(x_index_test, self.test_acc)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title('Accuracy of Training and Testing')
        plt.legend(['training', 'testing'], loc='upper left')
        plt.savefig('acc{}.png'.format(prefix))
