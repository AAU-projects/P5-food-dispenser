import os
import cv2
import numpy as np
from glob import glob

# Classes for training set
image_labels = ["cats", "dogs"]
# Image size convertion of training set
img_width_height = (128, 128)


def generate_labels():
    training_path = "data"
    data = []
    labels = []

    # Retreiving dataset for classes
    for x in range(0, len(image_labels)):
        dataset_path = os.path.join(os.getcwd(), training_path, image_labels[x])
        animals = glob(dataset_path + "/*.*")
        for animal in animals:
            # Reads image in greyscale and resizes
            image = cv2.imread(animal)
            image = cv2.resize(image, img_width_height)
            data.append(np.array(image))
            labels.append(x)
            if (len(data) % 500 == 0):
                print(len(data))

    # Converts and saves dataset
    animals = np.array(data)
    labels = np.array(labels)

    print("Saving results, this might take some time")
    np.save(os.path.join(training_path, "animals"), animals)
    np.save(os.path.join(training_path, "labels"), labels)


if __name__ == "__main__":
    generate_labels()