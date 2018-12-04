import os
from src.train_model import TrainModel
from src.image_processing import ImageProcessing

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.getcwd(), 'new data', 'dataset', 'pictures_dataset.npy')):
        print("[LOG] No npy files found... Generating")
        ImageProcessing().generate_labels()
           
    tm = TrainModel()
    tm.train_model()