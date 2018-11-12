import os
import sys
import cv2
import numpy as np
from glob import glob
from PIL import Image
from keras.utils import np_utils, to_categorical


class ImageProcessing:
    def __init__(self, *args, **kwargs):
        self.image_labels = ["cats", "dogs"]
        self.training_data_path= "data"
        self.img_width_height = (128, 128)
        self.generate_folders = ['test','dataset']
        
    def __load_image(self, filepath):
        img = cv2.imread(filepath)
        img_from_ar = Image.fromarray(img, 'RGB')
        resized_image = img_from_ar.resize(self.img_width_height)

        return np.array(resized_image)

    def __load_images(self, dataset_type, folder):
        print("[LOG] Loading " + dataset_type)
        folder_path = os.path.join(folder, self.training_data_path, dataset_type)
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
            dataset_path = os.path.join(os.getcwd(), training_path, self.image_labels[x])
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

    def retrieve_dataset(self, dataset_type, shuffle=True): 
        dataset_path = os.path.join(os.getcwd(), 'data', dataset_type)

        animals = np.load(os.path.join(dataset_path, f"animals_{dataset_type}.npy"))
        labels = np.load(os.path.join(dataset_path, f"labels_{dataset_type}.npy"))

        if shuffle:
            animals, labels = self.__shuffle(animals, labels)

        labels = animals.astype('float32')/255
        num_classes = len(self.image_labels)
        
        # One hot encoding
        labels = to_categorical(labels, num_classes)
        return animals, labels

    def retrieve_train_validation(self, shuffle=True, procent_split=0.9):
        animals, labels = retrieve_dataset("train", shuffle)

        train_size = len(animals) * procent_split

        animals_train, labels_train = animals[0:train_size], labels[0:train_size]
        animals_validation, labels_validation = animals[train_size:], labels[train_size:]

        return animals_train, labels_train, animals_validation, labels_validation

    def __shuffle(self, x, y):
        shape = np.arange(x.shape[0])
        np.random.shuffle(shape)
        x = x[shape]    
        y = y[shape]
        return x, y


if __name__ == "__main__":
    gen = ImageProcessing()
    gen.generate_labels(os.path.join( os.path.dirname( __file__ ), '..' ))
