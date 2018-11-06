import os
import cv2
import numpy as np
from glob import glob
from PIL import Image

# Classes for training set
image_labels = ["cats", "dogs"]

# Paths
training_data_path= "data"
training_folder = ["train", "validation"]

# Image size convertion of training set
img_width_height = (128, 128)

def load_image(filepath):
    img = cv2.imread(filepath)
    img_from_ar = Image.fromarray(img, 'RGB')
    resized_image = img_from_ar.resize(img_width_height)

    #image = cv2.imread(filepath)
    #image = cv2.resize(image, img_width_height)
    return np.array(resized_image)

def load_type(training_path):
    data = []
    labels = []

    # Retreiving dataset for classes
    for x in range(0, len(image_labels)):
        dataset_path = os.path.join(os.getcwd(), training_path, image_labels[x])
        animals = glob(dataset_path + "/*.*")
        for animal in animals:
            data.append(load_image(animal))
            labels.append(x)
            if (len(data) % 500 == 0):
                print(len(data))

    return data, labels

def load_images(foldertype):
    print("Loading " + foldertype)
    folder_path = os.path.join(training_data_path, foldertype)
    data, labels = load_type(folder_path)
    animals = np.array(data)
    labels = np.array(labels)

    print("Saving results, this might take some time")
    np.save(os.path.join(folder_path, "animals_" + foldertype), animals)
    np.save(os.path.join(folder_path, "labels_" + foldertype), labels)

def generate_labels():
    for i in range(0, len(training_folder)):
        load_images(training_folder[i])
        
if __name__ == "__main__":
    generate_labels()