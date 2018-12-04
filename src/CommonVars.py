import os

# Hidden vars

__picturePath = "new data"


# Static vars

classes = ["cats", "dogs", "junk"]
picture_folders = ['dataset', 'evalset']
img_hight, img_width = 128, 128


# Dynamic vars

picturePath = os.path.join(os.getcwd(),__picturePath)
num_classes = len(classes)
