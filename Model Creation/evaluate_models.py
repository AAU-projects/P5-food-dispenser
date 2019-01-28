import os
import keras
import sys
import src.CommonVars as vars
from glob import glob
from src.image_processing import ImageProcessing
from src.model_evaluate import ModelEvaluate
from keras.models import load_model


'''
Script to reevaluate all models in the model folder on a new evalset. 
The script renames each model to the new evaluvated accuracy score. 

HOW TO USE 
evaluate_models.py          - evaluates all models on the already generated evalset
evaluate_models.py 1        - creates new numpy files for evalset and evaluates all models in that evalset  
'''

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        # Generates a new evalset
        vars.picture_folders = ['evalset']
        ip = ImageProcessing()
        ip.generate_numpy_files()

    em = ModelEvaluate()

    # Iterates through each model
    subfolders = [f.name for f in os.scandir('models') if f.is_dir() ]  
    for y in range(0, len(subfolders)):
        path = f"models/{subfolders[y]}"
        model = keras.models.load_model(f"{path}/{subfolders[y]}.h5")
        score = em.evaluate_model(model, score_print=False)[1]
        score_rounded = f"{round(score, 3)"
        files = os.listdir(f"models/{subfolders[y]}/")
        split = subfolders[y].split('_')
        epochs = split[2]
        batch = split[3].strip()
        newname = f"model_{score_rounded}_{epochs}_{batch}"

        # Files renaming
        for x in range(0, len(files)):
            file_extension = os.path.splitext(files[x])[1]
            file_split = files[x].split('_')
            if len(file_split) >= 5:
                file_type = file_split[4]
                os.rename(f"{path}/{files[x]}", f"{path}/{newname}_{file_type}")
            else:
                os.rename(f"{path}/{files[x]}", f"{path}/{newname}.h5")
        
        # Folder files renaming
        os.rename(path, f"models/{newname}")
        
        # Log renaming
        logpath = os.path.join(os.getcwd(),'logs', subfolders[y])
        newlogpath = os.path.join(os.getcwd(),'logs', newname)
        os.rename(logpath, newlogpath)

        print(f"{subfolders[y]} -> {newname}")








