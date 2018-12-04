import os

# Hidden vars

__picturePath = "data"


# Static vars

classes = ["cats", "dogs"]
picture_folders = ['dataset', 'evalset']
img_height, img_width = 128, 128


# Dynamic vars

picturePath = os.path.join(os.getcwd(),__picturePath)
num_classes = len(classes)
