import numpy as np
from time import time
import matplotlib.pyplot as plt
import keras
import os
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from decimal import Decimal
from contextlib import redirect_stdout
from eval_model import evaluate_model
from src.image_processing import ImageProcessing 


"""
    def retrieve_generator(self, path, **kwargs):
        datagen = ImageDataGenerator(**kwargs)
        generator = datagen.flow_from_directory(path, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary', shuffle=True)
        return generator

    def retrieve_generators(self):
        train_generator = retrieve_generator(self.TRAIN_PATH, rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        validation_generator = retrieve_generator(self.VALIDATION_PATH, rescale=1. / 255)

        return train_generator, validation_generator
"""

def retrieve_generators():
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary', shuffle=True)
    validation_generator = test_datagen.flow_from_directory(VALIDATION_PATH, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary', shuffle=True)

    return train_generator, validation_generator

def retrive_dataset():
    img_processing = ImageProcessing()
    animals_train, labels_train, animals_validation, labels_validation = img_processing.retrieve_train_validation()

    global training_lenght
    global validation_lenght

    training_lenght = len(animals_train)
    validation_lenght = len(animals_validation)

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
    plt.savefig(model_path + "{}_graph.png".format(model_name))


def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))
    
    for name, history in histories:
      val = plt.plot(history.epoch, history.history['val_'+key],
                     '--', label=name.title()+' Val')
      plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
               label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()

def fit_model_generator(model):
    train_generator, validation_generator = retrieve_generators()
    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = validation_generator.n // validation_generator.batch_size
    return model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, 
                               validation_data=validation_generator, validation_steps=step_size_valid, epochs=epoch_size)

def fit_model_numpy(model, callbacks=None):
    animals_train, labels_train, animals_validation, labels_validation = retrive_dataset()
    history = model.fit(animals_train, labels_train, batch_size=batch_size, epochs=epoch_size, verbose=1, validation_data=(animals_validation, labels_validation),callbacks=callbacks)

    score = evaluate_model(model)
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
    with open(model_path + f"{model_name}_summary.txt", 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
        for x in range(0, len(activation_functions)):
            f.write(f"activation_{x + 1} = {activation_functions[x]}\n")
        
        f.write(f"Train set size: {training_lenght}\n")
        f.write(f"Validation set size: {validation_lenght}\n")


def train_model():
    model = create_model()

    #TensorBoard
    tensorboard_name = str(time())
    tensorboard = TensorBoard(log_dir="logs/{}".format(tensorboard_name));
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)

    history, score = fit_model_numpy(model, [tensorboard, earlyStop])

    plot_history([('model', history)])

    score_rounded = f"{round(score[1], 3):.3f}"
    model_name = get_model_name(score_rounded)
    model_path = "models/{}/".format(model_name)

    print("New model saved in " + model_path)
    create_model_folder(model_path)
    model.save(model_path + model_name + ".h5")

    save_model_graph(history, model_path, model_name)
    save_model_summary(model, model_path, model_name)
    rename_model_log(tensorboard_name, model_name)

def rename_model_log(tensorboard_name, model_name):
    logpath = os.path.join(os.getcwd(),'logs', tensorboard_name)
    newlogpath = os.path.join(os.getcwd(),'logs', model_name)

    if os.path.exists(logpath):
        os.rename(logpath, newlogpath)
        
if __name__ == "__main__":
    train_model()