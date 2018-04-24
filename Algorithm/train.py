from data_loader import data_loader
from models.networks import MLP
from sklearn import preprocessing


data = data_loader.DataLoader(False).load_data()

X = preprocessing.scale(data.training_dev)

batch_size = 200

nn = MLP([128, 512, 64, 10], dropouts=[0.5, 0.1, -1], activation='l_relu' )
loss = nn.fit(X, data.label_dev, data.training_val, data.label_val, my=0.95, learning_rate=1e-4,
              epochs=10000, batchsize=batch_size)

print("loss: {}".format(loss))
