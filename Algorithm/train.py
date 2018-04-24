from data_loader import data_loader
from models.networks import MLP
import numpy as np


data = data_loader.DataLoader(False).load_data()

mean = np.mean(data.training_dev, axis=0)
std = np.std(data.training_dev, axis=0)

X_train = data.training_dev - mean / std
X_val = data.training_val - mean / std

batch_size = 200

nn = MLP([128, 512, 64, 10], dropouts=[0.5, 0.1, -1], activation='l_relu' )
loss = nn.fit(X_train, data.label_dev, X_val, data.label_val, my=0.95, learning_rate=1e-4,
              epochs=500, batchsize=batch_size)

print("loss: {}".format(loss))
