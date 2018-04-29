import csv

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

nn = MLP([128, 512, 64, 10], dropouts=[0.5, 0.1, -1], activation='relu', norm="wn", update_type="nes_momentum")
train_acc, train_loss, test_acc, test_loss = nn.fit(X_train, data.label_dev, X_val, data.label_val, my=0.95,
                                                    learning_rate=1e-4,
                                                    epochs=max_epoch, batchsize=batch_size)

prefix = 'layer_3'
f = open('results_{}.csv'.format(prefix), "w")
field_names = ['train_acc', 'train_loss', 'test_acc', 'test_loss']
writer = csv.DictWriter(f,fieldnames=field_names)
writer.writeheader()
row = dict()

for i in range(len(train_acc)):
    row['train_acc'] = train_acc[i]
    row['train_loss'] = train_loss[i]
    row['test_acc'] = test_acc[i]
    row['test_loss'] = test_loss[i]
    writer.writerow(row)
f.close()

plt = Plot(max_epoch, train_acc, train_loss, test_acc, test_loss, prefix=prefix)
plt.plot()
