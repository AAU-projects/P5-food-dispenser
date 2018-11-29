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
        self.num_classes = 3

    def __load_image(self, filepath):
        img = cv2.imread(filepath)
        try:
            img_from_ar = Image.fromarray(img, 'RGB')
            resized_image = img_from_ar.resize(self.img_width_height)

            return np.array(resized_image)
        except AttributeError:
            print(f"Bad file: {filepath}")
            return None

    def __load_images(self, picture_folder, base_path):
        print("[LOG] Loading " + picture_folder)
        folder_path = os.path.join(base_path, self.training_data_path, picture_folder)
        data, labels = self.__load_type(folder_path)

        if len(data) > 0:
            pictures = np.array(data)
            labels = np.array(labels)

            print("[LOG] Saving results, this might take some time")
            np.save(os.path.join(folder_path, "pictures_" + picture_folder), pictures)
            np.save(os.path.join(folder_path, "labels_" + picture_folder), labels)

    def __load_type(self, training_path):
        data = []
        labels = []

        # Retreiving dataset for classes
        for x in range(0, len(self.image_labels)):
            dataset_path = os.path.join(
                os.getcwd(), training_path, self.image_labels[x])
            if not os.path.exists(dataset_path):
                print(f"[ERROR] Requires data folder at {dataset_path}")
                continue
            pictures = glob(dataset_path + "/*.*")
            for picture in pictures:
                img = self.__load_image(picture)
                if img is not None:
                    data.append(img)
                    labels.append(x)
                if (len(data) % 500 == 0):
                    print(f"[LOG] {len(data)}")

        return data, labels

    def generate_labels(self, path=os.getcwd()):
        for i in range(0, len(self.picture_folders)):
            self.__load_images(self.picture_folders[i], path)

    def retrive_dataset_old(self, folder):
        print("Loading images")
        pictures = np.load(os.path.join(self.training_data_path, folder, f"pictures_{folder}.npy"))
        labels = np.load(os.path.join(self.training_data_path, folder, f"labels_{folder}.npy"))

        return pictures, labels
        
    def retrive_dataset_test(self):
        print("Loading images")
        pictures, labels = self.retrive_dataset_old(self.picture_folders[1])
        pictures, labels = self.__shuffle(pictures, labels)
        pictures = pictures.astype('float32')/255

        # One hot encoding
        print("One hot encoding")
        labels = keras.utils.to_categorical(labels, self.num_classes)

        return pictures, labels
        
    def retrieve_train_validation(self, shuffle=True, procent_split=0.9):
        pictures, labels = self.retrive_dataset_old(self.picture_folders[0])
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