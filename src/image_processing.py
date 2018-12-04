import os
import sys
import cv2
import numpy as np
import keras
from glob import glob
from PIL import Image
from keras.utils import np_utils, to_categorical

class ImageProcessing:

    def __init__(self, *args, **kwargs):
        self.image_labels = ["cats", "dogs", "junk"]
        self.training_data_path = "new data"
        self.img_width_height = (128, 128)
        self.picture_folders = ['dataset', 'evalset']
        self.num_classes = len(self.image_labels)

    def load_image(self, filepath):
        img = cv2.imread(filepath)
        try:
            img_from_ar = Image.fromarray(img, 'RGB')
            resized_image = img_from_ar.resize(self.img_width_height)

            return np.array(resized_image)
        except AttributeError:
            print(f"Bad file: {filepath}")
            return None

    def save_images(self, classes, labels, path, dataset_type):
            np.save(os.path.join(path, "pictures_" + dataset_type), np.array(classes))
            np.save(os.path.join(path, "labels_" + dataset_type), np.array(labels))

    def load_picture_folders(self, dataset_type, folder):
        data_full = []
        labels_full = []
        print("[LOG] Loading " + dataset_type)

        folder_path = os.path.join(folder, self.training_data_path, dataset_type)

        for x in range(0, len(self.image_labels)):
            print("[LOG] Loading " + self.image_labels[x])
            data, labels, path = self.load_type(folder_path, self.image_labels[x], x)
            self.save_images(data, labels, path, dataset_type)
            data_full.extend(data)
            labels_full.extend(labels)

        if len(data) > 0:
            self.save_images(data_full, labels_full, folder_path, dataset_type)

    def load_type(self, training_path, image_label, class_value):
        # Retreiving dataset for classes
        data = []
        labels = []

        dataset_path = os.path.join(os.getcwd(), training_path, image_label)
        if os.path.exists(dataset_path):
            pictures = glob(dataset_path + "/*.jpg")
            pictures.extend(glob(dataset_path + "/*.png"))
            for picture in pictures:
                data.append(self.load_image(picture))
                labels.append(class_value)
                if (len(data) % 500 == 0):
                    print(f"[LOG] {len(data)}")
        else:
            print(f"[ERROR] Requires data folder at {dataset_path}")

        return data, labels, dataset_path

    def generate_labels(self, path=os.getcwd()):
        for i in range(0, len(self.picture_folders)):
            self.load_picture_folders(self.picture_folders[i], path)

    def retrive_dataset(self, datatype):
        print("Loading images")
        animals = np.load(f"{self.training_data_path}/{datatype}/pictures_{datatype}.npy")
        labels = np.load(f"{self.training_data_path}/{datatype}/labels_{datatype}.npy")

        return animals, labels

    def retrive_dataset_path(self, path, datatype):
        # print("Loading images")
        pictures = np.load(f"{path}/pictures_{datatype}.npy")
        labels = np.load(f"{path}/labels_{datatype}.npy")

        pictures, labels = self.__shuffle(pictures, labels)
        pictures = pictures.astype('float32')/255

        # One hot encoding
        # print("One hot encoding")
        labels = keras.utils.to_categorical(labels, self.num_classes)

        return pictures, labels

    def retrieve_train_validation(self, shuffle=True, procent_split=0.9):
        pictures, labels = self.retrive_dataset(self.picture_folders[0])
        pictures, labels = self.__shuffle(pictures, labels)
        train_size = int(len(pictures) * procent_split)

        # Get training set
        pictures_train = pictures[:train_size]
        labels_train = labels[:train_size]
        # Get validation set
        pictures_validation = pictures[train_size:]
        labels_validation = labels[train_size:]

        # Format picrures to right format
        pictures_train = pictures_train.astype('float32')/255
        pictures_validation = pictures_validation.astype('float32')/255

        # One hot encoding
        print("One hot encoding")
        labels_train = keras.utils.to_categorical(labels_train, self.num_classes)
        labels_validation = keras.utils.to_categorical(labels_validation, self.num_classes)

        return pictures_train, labels_train, pictures_validation, labels_validation

    def __shuffle(self, x, y):
        shape = np.arange(x.shape[0])
        np.random.shuffle(shape)
        x = x[shape]
        y = y[shape]
        return x, y

if __name__ == "__main__":
    gen = ImageProcessing()
    gen.generate_labels(os.path.join(os.path.dirname(__file__), '..'))
