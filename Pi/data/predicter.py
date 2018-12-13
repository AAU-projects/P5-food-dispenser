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

# Convert and resize an image into a numpy array
def convert_to_array(img):
    img = cv2.imread(img)
    img_from_ar = Image.fromarray(img, 'RGB')
    resized_image = img_from_ar.resize((128, 128))

    return np.array(resized_image)

# If prediction < THRESHOLD then disregard prediction 
def get_class_with_threshold(score):
    print(score[0])
    for x in range(0, len(score[0])):
        if score[0][x] > THRESHOLD:
            return x
    return -1

# Given a file and a model, use model to predict on the file
def predict_animal(file, model):
    # Get numpy array of resize image
    ar = convert_to_array(file)
    ar = ar/255
    a = np.array([ar]) # test with and without array in array
    score = model.predict(a, verbose=0)
    acc = np.max(score)
    for x in range(0, 3):
        print(round(score[0][x],10))
    return get_class_with_threshold(score)

# Predict on all images in a folder, return the the most commen prediction result from all predictions
def predict_folder(destination):
    predictions_array = []
    files = glob.glob(destination + "/*")
    for x in range(0, len(files)):
        predictions_array.append(predict_animal(files[x], model))

    return most_common_prediction(predictions_array)

# Predict one image
def predict_file(file):
    return predict_animal(file, model)

# Return most common prediction 
def most_common_prediction(lst):
    return max(set(lst), key=lst.count)

# if the file is run directly, predict on the given file from cmd arguments
if __name__ == "__main__":
    print(predict_file(sys.argv[1]))
