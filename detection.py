import dataset
import fullyconn
import numpy as np
import math
from PIL import Image

dataset = dataset.load_dataset()

def train_autoencoder(dataset, batch_size, learning_rate, steps):
    nn = fullyconn.MLP(400)
    nn.add_layer("bottleneck", 200)
    nn.add_layer("output", 400)
    train = nn.build_train(learning_rate)

    batch_count = len(dataset) / batch_size
    for _ in range(0, steps):
        for i in range(1, batch_count):
            ts = dataset[i * batch_size: (i + 1) * batch_size]
            print(math.sqrt(train(ts, ts) / 400))

    return nn

batch_size = 100
learning_rate = 0.01
steps = 1

aa = train_autoencoder(dataset, batch_size, learning_rate, steps)
eval = aa.build_eval()

a = dataset[0]
b = eval(np.array([a]))
a = np.array(a.reshape(20, 20) * 255, dtype="uint8")
b = np.array(b.reshape(20, 20) * 255, dtype="uint8")

im  = Image.fromarray(a)
im2 = Image.fromarray(b)
imrgb = Image.merge("RGB", (im, im, im))
imrgb2 = Image.merge("RGB", (im2, im2, im2))

imrgb.save("original.png")
imrgb2.save("reconstructed.png")
