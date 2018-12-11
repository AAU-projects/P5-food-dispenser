import os
import src.CommonVars as vars
from src.train_model import TrainModel
from src.image_processing import ImageProcessing

'''
Script to evaluate a model given in args on each class which prints the accuracy for each class. 

HOW TO USE 
evaluate_classes.py model_0.874_100_64        - evaluates model_0.874_100_64 on the classes defined in CommonVars
'''

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.getcwd(), vars.picturePath, vars.picture_folders[0], 'pictures_dataset.npy')):
        print("[LOG] No npy files found... Generating")
        ImageProcessing().generate_numpy_files()
           
    tm = TrainModel()
    tm.train_model()