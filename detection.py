import dataset
import autoencoder

dataset = dataset.load_dataset()

a = autoencoder.Autoencoder()
a.train(dataset[0:300], 10, 1000, 0.0001)
