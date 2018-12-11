import io
import numpy as np
from src.image_processing import ImageProcessing
import src.CommonVars as vars

class ModelEvaluate:
    # Evaluate a given model from a given path    
    def evaluate_model(self, model, path=f"{vars.picturePath}/{vars.picture_folders[1]}", score_print=True):
        img_processing = ImageProcessing()

        # Retrieve evalset
        data, labels = img_processing.retrive_dataset_path(path, vars.picture_folders[1])

        # Evaluate model
        score = model.evaluate(data, labels, verbose=score_print)
        
        if(score_print):
            self.__print_score(score[1])
        return score

    # Print out the final model accuracy 
    def __print_score(self, score):
        print(f"\n\n{'#'*20}\n\n[LOG] Model accuracy: {score * 100}%\n")

        progressbar_size = 40
        score_value = score * progressbar_size
        score_rest = progressbar_size - score_value
        print(f"|{'█'* int(score_value)}{'░'* int(score_rest)}|")

        print(f"\n\n{'#'*20}")
