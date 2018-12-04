import os
import sys
import src.CommonVars as vars
from glob import glob
from src.model_evaluate import ModelEvaluate

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        model_name = sys.argv[1]
        model_path = f"models/{model_name}/{model_name}.h5"

        if os.path.exists(model_path):
            import keras
            from keras.models import load_model
            model = keras.models.load_model(model_path)
            me = ModelEvaluate()

            score = me.evaluate_model(model, f"{vars.picturePath}/{vars.picture_folders[1]}/", True)
            score_rounded = f"{round(score[1], 5):.5f}"
            print(f"All {score_rounded}")
            sum = 0.0
            for x in range(0, vars.num_classes):
                score = me.evaluate_model(model, f"{vars.picturePath}/{vars.picture_folders[1]}/{vars.classes[x]}", False)
                score_rounded = f"{round(score[1], 5):.5f}"
                sum += float(score_rounded)
                print(f"{vars.classes[x]} {score_rounded}")
            print(f"Average {sum/vars.num_classes}")
        else:
            print('Model not found')
    else:
        print('Model input required')
