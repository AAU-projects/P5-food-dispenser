import os
import cv2
import numpy as np
from glob import glob

# Classes for training set
image_labels = ["cats", "dogs"]
# Image size convertion of training set
img_width_height = (150, 150)

training_path = "data/"
data = []
labels = []

# Retreiving dataset for classes
for x in range(0, len(image_labels)):
  animals = glob(os.path.join(os.getcwd(), training_path, image_labels[x]) + "/*.")
  for animal in animals:
      # Reads image in greyscale and resizes
      image = cv2.imread(animal)
      image = cv2.resize(image, img_width_height)
      data.append(image)
      labels.append(x)

# Converts and saves dataset
animals = np.array(data)
labels = np.array(labels)

np.save(os.path.join(training_path, "animals"), animals)
np.save(os.path.join(training_path, "labels"), labels)