import os
import sys
import src.CommonVars as vars
from glob import glob
from src.model_evaluate import ModelEvaluate

'''
Script to evaluate a model given in args on each class which prints the accuracy for each class. 

HOW TO USE 
evaluate_classes.py model_0.874_100_64        - evaluates model_0.874_100_64 on the classes defined in CommonVars
'''
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        model_name = sys.argv[1]
        model_path = f"models/{model_name}/{model_name}.h5"

        if os.path.exists(model_path):
            import keras
            from keras.models import load_model
            model = keras.models.load_model(model_path)
            me = ModelEvaluate()

            picture_path = os.path.join(vars.picturePath, vars.picture_folders[1])
            score = me.evaluate_model(model, picture_path, True)
            score_rounded = f"{round(score[1], 5)"
            print(f"All {score_rounded}")
            sum = 0.0
            for x in range(0, vars.num_classes):
                score = me.evaluate_model(model, os.path.join(picture_path, vars.classes[x]), False)
                score_rounded = f"{round(score[1], 5)"
                sum += float(score_rounded)
                print(f"{vars.classes[x]} {score_rounded}")
            print(f"[LOG] Average {sum/vars.num_classes}")
        else:
            print('[ERROR] Model not found')
    else:
        print('[ERROR] Model input required')