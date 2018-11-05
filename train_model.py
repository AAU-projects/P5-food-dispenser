import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from keras import backend as K

epoch_size = 10 # total number of runs
batch_size = 16 # parts to split dataset into
TRAIN_PATH = 'data/train'
VALIDATION_PATH = 'data/validation'

img_width, img_height = 128, 128

def create_model():    
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model    

def retrieve_generators():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(VALIDATION_PATH, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

    return train_generator, validation_generator

def plot_model_training(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def train_model():
    model = create_model()
    train_generator, validation_generator = retrieve_generators()

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = validation_generator.n // validation_generator.batch_size

    history = model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, validation_data=validation_generator, validation_steps=step_size_valid, epochs=epoch_size)
    model.save("model.h5")
    plot_model_training(history)

if __name__ == "__main__":
    train_model()