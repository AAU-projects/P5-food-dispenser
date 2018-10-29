#!/usr/bin/env python
# - *- coding: utf- 8 - *-
from keras.models import load_model
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import time
import sys
import glob

img_width, img_height = 100, 100
model = load_model('Model/Model.h5')

def convert_to_array(img):
    image = np.array(Image.open(img), dtype=np.uint8)
    img_from_ar = Image.fromarray(image, 'RGB')
    resized_image = img_from_ar.resize((img_width, img_height))
    return np.array(resized_image)

def get_animal_name(score):
    print(score[0])
    if score[0][0] > 0.8:
        return 0
    if score[0][0] < 0.2:
        return 1
    return -1


def predict_animal(file, model):
    ar = convert_to_array(file)
    ar = ar/255
    label = 1
    a = []
    a.append(ar)
    a = np.array(a)
    score = model.predict(a, verbose=1)
    acc = np.max(score)
    return get_animal_name(score)


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
