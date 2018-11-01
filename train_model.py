import numpy as np
from math import ceil
#from keras.models import Sequential

TEST_RATIO = 0.1

def retrieve_dataset():
    # Loads dataset
    animals = np.load("data/animals.npy")
    labels = np.load("data/labels.npy")

    # Shuffles dataset
    shape = np.arange(animals.shape[0])
    np.random.shuffle(shape)
    animals = animals[shape]
    labels = labels[shape]
    return animals, labels, shape

def retrieve_training_test_dataset(animals, labels):
    dataset_size = len(animals)
    
    # Splits into test set
    x_test = animals[0:ceil(dataset_size*TEST_RATIO)]
    y_test = labels[0:ceil(dataset_size*TEST_RATIO)]

    # Splits into train set
    x_train = animals[ceil(dataset_size*TEST_RATIO):]
    y_train = labels[ceil(dataset_size*TEST_RATIO):]

def train_model():
    animals, labels, shape = retrieve_dataset()
    retrieve_training_test_dataset(animals, labels)

if __name__ == "__main__":
    train_model()