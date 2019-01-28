import os
import sys
import cv2
import numpy as np
import keras
import src.CommonVars as vars
from glob import glob
from PIL import Image
from keras.utils import np_utils, to_categorical

class ImageProcessing:
    def load_image(self, filepath):
        # Load picture from filepath
        img = cv2.imread(filepath)
        # Resize image and return as an array
        try:
            img_from_ar = Image.fromarray(img, 'RGB')
            resized_image = img_from_ar.resize((vars.img_width, vars.img_height))

            return np.array(resized_image)

        # File could not be converted return nothing
        except AttributeError:
            print(f"[ERROR] Bad file: {filepath}")

            return None

    # Saves the images with a corresponding labels file that matches the position of the images
    def save_images(self, classes, labels, path, dataset_type):
            np.save(os.path.join(path, "pictures_" + dataset_type), np.array(classes))
            np.save(os.path.join(path, "labels_" + dataset_type), np.array(labels))

    # Loads all classes in a dataset type and saves them
    def load_class_folders(self, dataset_type, folder):
        data_full = []
        labels_full = []
        print("[LOG] Loading " + dataset_type)

        folder_path = os.path.join(folder, vars.picturePath, dataset_type)

        # Iterates through each class and saves class images
        for x in range(0, len(vars.classes)):
            print("[LOG] Loading " + vars.classes[x])
            data, labels, path = self.load_class_pictures(folder_path, vars.classes[x], x)
            self.save_images(data, labels, path, dataset_type)
            data_full.extend(data)
            labels_full.extend(labels)

        # Saves the compelte dataset
        if len(data) > 0:
            self.save_images(data_full, labels_full, folder_path, dataset_type)

    # Loads all pictures in a class
    # training_path path to the dataset class folder
    # image_label is the class, i.e cat, dog, junk
    # class_value is the label value that describes the class
    def load_class_pictures(self, training_path, image_label, class_value):
        data = []
        labels = []

        dataset_path = os.path.join(os.getcwd(), training_path, image_label)
        if os.path.exists(dataset_path):
            pictures = glob(dataset_path + "/*.jpg")            
            pictures.extend(glob(dataset_path + "/*.png"))      # Retrives all .jpg and png images in dataset_path
            for picture in pictures:
                data.append(self.load_image(picture))           # Appends each image to the data array
                labels.append(class_value)                      # Appends a class_value to each labels that corresponds to the image class
                if (len(data) % 500 == 0):
                    print(f"[LOG] {len(data)}")
        else:
            print(f"[ERROR] Requires data folder at {dataset_path}")

        return data, labels, dataset_path

    # Generates numpy files for all datasets
    def generate_numpy_files(self, path=os.getcwd()):
        for i in range(0, len(vars.picture_folders)):
            self.load_class_folders(vars.picture_folders[i], path)

    # Retrieves the numpy files for the given dataset type
    def retrive_dataset(self, settype):
        print("[LOG] Loading images")
        animals = np.load(f"{vars.picturePath}/{settype}/pictures_{settype}.npy")
        labels = np.load(f"{vars.picturePath}/{settype}/labels_{settype}.npy")

        return animals, labels

    # Retrieves a numpy dataset
    def retrive_dataset_path(self, path, datatype):
        # Loading images
        pictures = np.load(f"{path}/pictures_{datatype}.npy")
        labels = np.load(f"{path}/labels_{datatype}.npy")

        # Shuffles dataset        
        pictures, labels = self.__shuffle(pictures, labels)
        # /255 Normalizer so that all pixels has a value between 0 - 1
        pictures = pictures.astype('float32')/255

        # One hot encoding
        labels = keras.utils.to_categorical(labels, vars.num_classes)

        return pictures, labels

    # Retrieves the dataset and splits it into a training and validation set    
    def retrieve_train_validation(self, shuffle=True, procent_split=0.9):
        pictures, labels = self.retrive_dataset(vars.picture_folders[0])
        pictures, labels = self.__shuffle(pictures, labels)
        train_size = int(len(pictures) * procent_split)

        # Get training set from 0 - 0.9
        pictures_train = pictures[:train_size]
        labels_train = labels[:train_size]

        # Get validation set from 0.9 to 1
        pictures_validation = pictures[train_size:]
        labels_validation = labels[train_size:]

        # Format picrures to right format
        # /255 Normalizer so that all pixels has a value between 0 - 1
        pictures_train = pictures_train.astype('float32')/255
        pictures_validation = pictures_validation.astype('float32')/255

        # One hot encoding
        print("[LOG] One hot encoding")
        labels_train = keras.utils.to_categorical(labels_train, vars.num_classes)
        labels_validation = keras.utils.to_categorical(labels_validation, vars.num_classes)

        return pictures_train, labels_train, pictures_validation, labels_validation

    # Shuffles a dataset randomly    
    def __shuffle(self, x, y):
        shape = np.arange(x.shape[0])
        np.random.shuffle(shape)
        x = x[shape]
        y = y[shape]
        return x, y

if __name__ == "__main__":
    gen = ImageProcessing()
    gen.generate_numpy_files(os.path.join(os.path.dirname(__file__), '..'))
