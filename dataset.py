from scipy import misc
import numpy as np
import os, sys
import pickle

def progress(task, progress):
    sys.stdout.write('\r{0}: [{1}] {2}%'.format(task, '#'*(progress/10) + ' '*(10 - progress/10), progress))
    if (progress == 100):
        sys.stdout.write("\n")
    sys.stdout.flush()

def get_training_files():
    res = []
    for root, sub, files in os.walk("dataset"):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension == ".jpg":
                res.append(os.path.join(root, file))
    return res

def generate_dataset():
    tf = get_training_files()
    dataset = np.empty([len(tf), 400], np.dtype("u1"))
    for i, image_file in enumerate(tf):
        progress("Loading pictures", int(float(i + 1) / len(tf) * 100))
        im = misc.imresize(misc.imread(image_file), (20, 20))
        for (x, y) in np.ndindex(20, 20):
            dataset[i][x + y * 20] = np.dot(im[x][y], np.array([0.21, 0.71, 0.07]))
    f = open("dataset.bin", "w")
    pickle.dump(dataset, f)
    return dataset

def load_dataset():
    if os.path.isfile("dataset.bin"):
        print("Loading dataset.bin")
        f = open("dataset.bin", "r")
        dataset = pickle.load(f)
    else:
        print("Generating dataset.bin")
        dataset = generate_dataset()

    return dataset

dataset = load_dataset()
for im in dataset:
    print(im.shape)
print("Let's play ?")
