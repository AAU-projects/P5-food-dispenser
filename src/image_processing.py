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
        self.image_labels = ["cats", "dogs"]
        self.training_data_path = "data"
        self.img_width_height = (128, 128)
        self.generate_folders = ['dataset', 'test']

    def __load_image(self, filepath):
        img = cv2.imread(filepath)
        img_from_ar = Image.fromarray(img, 'RGB')
        resized_image = img_from_ar.resize(self.img_width_height)

        return np.array(resized_image)

    def __load_images(self, dataset_type, folder):
        print("[LOG] Loading " + dataset_type)
        folder_path = os.path.join(
            folder, self.training_data_path, dataset_type)
        data, labels = self.__load_type(folder_path)

        if len(data) > 0:
            animals = np.array(data)
            labels = np.array(labels)

            print("[LOG] Saving results, this might take some time")
            np.save(os.path.join(folder_path, "animals_" + dataset_type), animals)
            np.save(os.path.join(folder_path, "labels_" + dataset_type), labels)

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
            animals = glob(dataset_path + "/*.*")
            for animal in animals:
                data.append(self.__load_image(animal))
                labels.append(x)
                if (len(data) % 500 == 0):
                    print(f"[LOG] {len(data)}")

        return data, labels

    def generate_labels(self, path=os.getcwd()):
        for i in range(0, len(self.generate_folders)):
            self.__load_images(self.generate_folders[i], path)


    def retrive_dataset_old(self, datatype):
        print("Loading animals")
        animals = np.load(f"data/{datatype}/animals_{datatype}.npy")
        labels = np.load(f"data/{datatype}/labels_{datatype}.npy")

        return animals, labels
        
    def retrive_dataset_test(self):
        print("Loading animals")
        animals, labels = self.retrive_dataset_old("test")
        animals, labels = self.__shuffle(animals, labels)
        animals = animals.astype('float32')/255

        num_classes = 2
        # One hot encoding
        print("One hot encoding")
        labels = keras.utils.to_categorical(labels, num_classes)

        return animals, labels
        

    def retrieve_train_validation(self, shuffle=True, procent_split=0.9):
        animals, labels = self.retrive_dataset_old("dataset")

        animals, labels = self.__shuffle(animals, labels)

        train_size = int(len(animals) * procent_split)
        animals_train = animals[:train_size]
        labels_train = labels[:train_size]
        animals_validation = animals[train_size:]
        labels_validation = labels[train_size:]

        animals_train = animals_train.astype('float32')/255
        animals_validation = animals_validation.astype('float32')/255

        num_classes = 2
        # One hot encoding
        print("One hot encoding")
        labels_train = keras.utils.to_categorical(labels_train, num_classes)
        labels_validation = keras.utils.to_categorical(labels_validation, num_classes)

        return animals_train, labels_train, animals_validation, labels_validation

    def __shuffle(self, x, y):
        shape = np.arange(x.shape[0])
        np.random.shuffle(shape)
        x = x[shape]
        y = y[shape]
        return x, y


if __name__ == "__main__":
    gen = ImageProcessing()
    gen.generate_labels(os.path.join(os.path.dirname(__file__), '..'))
