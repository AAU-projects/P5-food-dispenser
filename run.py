import os
from src.train_model import TrainModel
from src.image_processing import ImageProcessing

if __name__ == "__main__":
    if not os.path.exists(f"{os.getcwd()}/data/dataset/animals_dataset.npy"):
        print("[LOG] No npy files found... Generating")
        ip = ImageProcessing()
        ip.generate_labels()
    tm = TrainModel()
    tm.train_model()
