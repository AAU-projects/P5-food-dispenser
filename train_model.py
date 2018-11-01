import numpy as np
from math import ceil
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense

from generate_labels import img_width_height as imgsize

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

    return x_train, y_train, x_test, y_test

def create_model(shape):    
    model = Sequential()

    #https://keras.io/layers/convolutional/#conv2d
    model.add(Conv2D(imgsize[0], (3,3), padding='same', activation='relu', input_shape=shape))

    return model


def train_model():
    animals, labels, shape = retrieve_dataset()
    model = create_model(shape)
    x_train, y_train, x_test, y_test = retrieve_training_test_dataset(animals, labels)



if __name__ == "__main__":
    train_model()