import io
import numpy as np
from image_processing import ImageProcessing

class ModelEval:
    
    @classmethod
    def evaluate_model(self, model, score_print=True):
        img_processing = ImageProcessing()
        data, labels = img_processing.retrieve_dataset("test")
        score = model.evaluate(data, labels, verbose=1)
        if(score_print):
            self.__print_score(score[1])
        return score

    def __print_score(self, score):
        print(f"\n\n{'#'*20}\n\nModel accuracy: {score * 100}%\n")

        progressbar_size = 40
        score_value = score * progressbar_size
        score_rest = progressbar_size - score_value
        print(f"|{'█'* int(score_value)}{'░'* int(score_rest)}|")
    
        print(f"\n\n{'#'*20}")