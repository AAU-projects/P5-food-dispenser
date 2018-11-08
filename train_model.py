import numpy as np
from time import time
import matplotlib.pyplot as plt
import keras
import os
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from decimal import Decimal

epoch_size = 2 # total number of runs
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
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model    

def retrieve_generators():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary', shuffle=True)
    validation_generator = test_datagen.flow_from_directory(VALIDATION_PATH, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary', shuffle=True)

    return train_generator, validation_generator

def shuffle(x, y):
    shape = np.arange(x.shape[0])
    np.random.shuffle(shape)
    x = x[shape]
    y = y[shape]
    return x, y

def retrive_dataset():
    animals_train = np.load("data/train/animals_train.npy")
    labels_train = np.load("data/train/labels_train.npy")
    animals_train, labels_train = shuffle(animals_train, labels_train)
    animals_train = animals_train.astype('float32')/255

    animals_validation = np.load("data/validation/animals_validation.npy")
    labels_validation = np.load("data/validation/labels_validation.npy")
    animals_validation, labels_validation = shuffle(animals_validation, labels_validation)
    animals_validation = animals_validation.astype('float32')/255

    num_classes = 2
    # One hot encoding
    labels_train = keras.utils.to_categorical(labels_train, num_classes)
    labels_validation = keras.utils.to_categorical(labels_validation,num_classes)

    return animals_train, labels_train, animals_validation, labels_validation

def save_model_graph(history, model_path, model_name):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.title('Model accuracy, epochs: {}, batch: {}'.format(epoch_size, batch_size))
    plt.ylabel('Accuracy/Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Test_acc', "Train_loss", "Test_loss"], loc='upper left')
    plt.savefig(model_path + "graph_{}.png".format(model_name))

def fit_model_generator(model):
    train_generator, validation_generator = retrieve_generators()
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = validation_generator.n // validation_generator.batch_size
    return model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, 
                               validation_data=validation_generator, validation_steps=step_size_valid, epochs=epoch_size)

def fit_model_numpy(model, callbacks=None):
    animals_train, labels_train, animals_validation, labels_validation = retrive_dataset()
    history = model.fit(animals_train, labels_train, batch_size=batch_size, epochs=epoch_size, verbose=1, validation_data=(animals_validation, labels_validation),callbacks=callbacks)
    score = model.evaluate(animals_validation, labels_validation, verbose=1)
    return history, score

def create_model_folder(path):
    if not os.path.exists("models"):
        os.makedirs("models")

    os.makedirs(path)

def get_model_name(score):
    save_value = 1

    model_name = "model_{}_{}_{}".format(str(score), epoch_size, batch_size)
    new_model_name = model_name

    while(os.path.exists("models/" + new_model_name)):
        new_model_name = model_name + "({})".format(save_value)
        save_value += 1
        
    return new_model_name

def save_model_summary(model, model_path, model_name):
    # Save summary.
    summary = model.summary()
    summary_file = open(model_path + f"{model_name}_summary.txt", 'w+')
    summary_file.write(summary)
    summary_file.close()

def train_model():
    model = create_model()

    #TensorBoard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()));

    history, score = fit_model_numpy(model, [tensorboard])

    score_rounded = round(score[1], 2)
    model_name = get_model_name(score_rounded)
    model_path = "models/{}/".format(model_name)

    print("New model saved in " + model_path)
    create_model_folder(model_path)
    model.save(model_path + model_name + ".h5")

    save_model_graph(history, model_path, model_name)
    save_model_summary(model, model_path, model_name)


if __name__ == "__main__":
    train_model()