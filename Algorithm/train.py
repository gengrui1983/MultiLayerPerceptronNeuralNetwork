import csv
import h5py

from data_loader import data_loader
from models.networks import MLP
import numpy as np

from models.plot import Plot
import timeit

t1 = timeit.default_timer()

data = data_loader.DataLoader(False).load_data()

mean = np.mean(data.training_dev, axis=0)
std = np.std(data.training_dev, axis=0)

X_train = data.training_dev - mean / std
X_val = data.training_val - mean / std
X_test = data.testing - mean / std

batch_size = 200
max_epoch = 1000

nn = MLP([128, 512, 256, 64, 10], dropouts=[-1, -1, -1, -1], activation='relu', norm="wn", update_type="nes_momentum")
train_acc, train_loss, test_acc, test_loss = nn.fit(X_train, data.label_dev, X_val, data.label_val, my=0.95,
                                                    learning_rate=1e-5,
                                                    epochs=max_epoch, batchsize=batch_size)
_, output = nn.predict(X_test)

with h5py.File('../Output/Predicted_labels.h5', 'w') as hf:
    hf.create_dataset("label",  data=output)

prefix = 'final_result'
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

t2 = timeit.default_timer()

print("running time is: {}".format(t2 - t1))
