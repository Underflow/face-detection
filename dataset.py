from scipy import misc
import numpy as np
import os, sys
import pickle
import theano

def progress(task, task_id, nb_task):
    progress = int(float(task_id) / nb_task * 100)
    sys.stdout.write('\r{0}: [{1}] {2}%'.format(task, '#'*(progress/10) + ' '*(10 - progress/10), progress))
    if (task_id == nb_task):
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
    tf = get_training_files()[:4000]
    dataset = np.empty([len(tf), 40, 40, 3], dtype=theano.config.floatX)
    for i, image_file in enumerate(tf):
        progress("Loading pictures", i, len(tf))
        dataset[i] = misc.imresize(misc.imread(image_file)[50:-50,50:-50], (40, 40)) / 255.
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
