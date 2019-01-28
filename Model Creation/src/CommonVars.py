import os
from keras.callbacks import TensorBoard, EarlyStopping
from time import time

'''
These variables was often used in the different classes
and instead of declaring them in each class, they were
all declared in here. This is also good for changing variables, 
and results in a more flexible code.
'''

# Hidden vars
__picturePath = "new data"

# Static vars
classes = ["cats", "dogs", "junk"]
picture_folders = ['dataset', 'evalset']
img_height, img_width = 128, 128

# Dynamic vars
picturePath = os.path.join(os.getcwd(),__picturePath)
num_classes = len(classes)

# Callbacks
tensorboard_name = str(time())
__tensorboard = TensorBoard(log_dir="logs/{}".format(tensorboard_name))
__earlyStop = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)
callbacks = [__tensorboard]
