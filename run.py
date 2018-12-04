import os
import src.CommonVars as vars
from src.train_model import TrainModel
from src.image_processing import ImageProcessing

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.getcwd(), vars.picturePath, vars.picture_folders[0], 'pictures_dataset.npy')):
        print("[LOG] No npy files found... Generating")
        ImageProcessing().generate_labels()
           
    tm = TrainModel()
    tm.train_model()