import dataset
import autoencoder

dataset = dataset.load_dataset()

a = autoencoder.Autoencoder()
a.train(dataset[0:30], 200, 0, 0.01)
while 1:
    for i in range(1,30):
        ts = dataset[i * 100: i * 100 + 100]
        autoencoder.Neuralnet.train(a, ts, ts, 20, 0.001)
