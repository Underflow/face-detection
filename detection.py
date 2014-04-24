import dataset
import autoencoder
import numpy as np
from PIL import Image

dataset = dataset.load_dataset()



def train_autoencoder(dataset, batch_size, learning_rate, steps):
    a = autoencoder.Autoencoder()
    a.train(dataset[0:batch_size], 375, 0, 0.3)

    batch_count = len(dataset) / batch_size
    for _ in range(1, steps):
        for i in range(1, batch_count):
            ts = dataset[i * batch_size: (i + 1) * batch_size]
            autoencoder.Neuralnet.train(a, ts, ts, steps, learning_rate)

    return a

batch_size = 40
learning_rate = 3.0 / batch_size
steps = 1

aa = train_autoencoder(dataset, batch_size, learning_rate, steps)

# Visualisation of the output layer of the autoencoder
a = dataset[1]
b = aa.eval(dataset[1])
a = np.array(a.reshape(20, 20) * 255, dtype="uint8")
b = np.array(b.reshape(20, 20) * 255, dtype="uint8")
im = Image.fromarray(a)
im2 = Image.fromarray(b)
imrgb = Image.merge("RGB", (im, im, im))
imrgb2 = Image.merge("RGB", (im2, im2, im2))
imrgb.show()
imrgb2.show()
