from data_loader import data_loader
from models.networks import MLP
import numpy as np

from models.plot import Plot

data = data_loader.DataLoader(False).load_data()

mean = np.mean(data.training_dev, axis=0)
std = np.std(data.training_dev, axis=0)

X_train = data.training_dev - mean / std
X_val = data.training_val - mean / std

batch_size = 200
max_epoch = 300

nn = MLP([128, 512, 64, 10], dropouts=[0.5, 0.1, -1], activation='relu', norm=None, update_type=None)
train_acc, train_loss, test_acc, test_loss = nn.fit(X_train, data.label_dev, X_val, data.label_val, my=0.95,
                                                    learning_rate=1e-4,
                                                    epochs=max_epoch, batchsize=batch_size)

plt = Plot(max_epoch, train_acc, train_loss, test_acc, test_loss, prefix="n_n_4")
plt.plot()
