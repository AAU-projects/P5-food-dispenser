import numpy as np
import io
from keras.utils import np_utils, to_categorical
from keras.models import Sequential
from contextlib import redirect_stdout

def shuffle(x, y):
    shape = np.arange(x.shape[0])
    np.random.shuffle(shape)
    x = x[shape]    
    y = y[shape]
    return x, y 

def retrieve_test_dataset(): 
    animals_test = np.load("data/test/animals_test.npy")
    labels_test = np.load("data/test/labels_test.npy")

    animals_test, labels_test = shuffle(animals_test, labels_test)
    animals_test = animals_test.astype('float32')/255

    num_classes = 2
    # One hot encoding
    labels_test = to_categorical(labels_test, num_classes)
    return animals_test, labels_test

def evaluate_model(model, score_print=True):
    data, labels = retrieve_test_dataset()
    f = io.StringIO()
    with redirect_stdout(f):
        score = model.evaluate(data, labels, verbose=1)
    if(score_print):  
        #print(model.metrics_names)
        #print(score)
        print_score(score[1])
    return score

def print_score(score):
    print(f"\n\n{'#'*20}\n\nModel accuracy: {score * 100}%\n")

    progressbar_size = 40
    score_value = score * progressbar_size
    score_rest = progressbar_size - score_value
    print(f"|{'█'* int(score_value)}{'░'* int(score_rest)}|")
    
    print(f"\n\n{'#'*20}")

if __name__ == "__main__":
    print_score(0.5)                                                                                                   