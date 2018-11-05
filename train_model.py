import numpy as np
from math import ceil
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from generate_labels import img_width_height as imgsize

TEST_RATIO = 0.1
EPOCH_SIZE = 50 # total number of runs
BATCH_SIZE = 16 # parts to split dataset into

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
    model.add(Conv2D(imgsize[0], (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(imgsize[0], (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(imgsize[0] * 2, (3, 3), activation='relu'))
    model.add(Conv2D(imgsize[0] * 2, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(imgsize[0] * 2, (3, 3), activation='relu'))
    model.add(Conv2D(imgsize[0] * 2, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(imgsize[0] * 2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    return model

def train_model():
    animals, labels, shape = retrieve_dataset()
    model = create_model(shape)
    
    x_train, y_train, x_test, y_test = retrieve_training_test_dataset(animals, labels)

    model.fit(x_train, y_train,
            epochs=EPOCH_SIZE,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            verbose=1)

    model.save("model.h5")

if __name__ == "__main__":
    train_model()