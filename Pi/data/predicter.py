#!/usr/bin/env python
# - *- coding: utf- 8 - *-
from keras.models import load_model
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import time
import sys
import cv2
import glob

model = load_model('model/model.h5')
THRESHOLD = 0.8

def convert_to_array(img):
    img = cv2.imread(img)
    img_from_ar = Image.fromarray(img, 'RGB')
    resized_image = img_from_ar.resize((128, 128))

    return np.array(resized_image)

def get_class_with_threshold(score):
    print(score[0])
    for x in range(0, len(score)):
        if score[0][x] > THRESHOLD:
            return x
    return -1

def predict_animal(file, model):
    ar = convert_to_array(file)
    ar = ar/255
    a = []
    a.append(ar)
    a = np.array(a)
    score = model.predict(a, verbose=0)
    acc = np.max(score)
    for x in range(0, 3):
        print(round(score[0][x],10))
    return get_class_with_threshold(score)

def predict_folder(destination):
    predictions_array = []
    files = glob.glob(destination + "/*")
    for x in range(0, len(files)):
        predictions_array.append(predict_animal(files[x], model))

    return most_common(predictions_array)

def predict_file(file):
    return predict_animal(file, model)

def most_common(lst):
    return max(set(lst), key=lst.count)

if __name__ == "__main__":
    print(predict_file(sys.argv[1]))
