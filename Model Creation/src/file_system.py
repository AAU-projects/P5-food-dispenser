import os
from contextlib import redirect_stdout

class FileSystem:
     # Create a folder for all trained models
    @classmethod
    def create_model_folder(cls, path):
        if not os.path.exists("models"):
            os.makedirs("models")

        os.makedirs(path)

    # Renames the tensorboard log to the score of the model
    @classmethod
    def rename_model_log(cls, tensorboard_name, model_name):
        logpath = os.path.join(os.getcwd(),'logs', tensorboard_name)
        newlogpath = os.path.join(os.getcwd(),'logs', model_name)

        if os.path.exists(logpath):
            os.rename(logpath, newlogpath)

    # Generates a model name for a model
    @classmethod        
    def generate_model_name(cls, score, epoch_size, batch_size):
        save_value = 1

        model_name = f"model_{str(score)}_{epoch_size}_{batch_size}"
        new_model_name = model_name

        # If an equal model exists 
        while(os.path.exists("models/" + new_model_name)):
            new_model_name = model_name + f"({save_value})"
            save_value += 1
            
        return new_model_name

    # Save a summary of the model
    @classmethod
    def save_model_summary(cls, model, model_path, model_name, training_lenght, validation_lenght):
        with open(model_path + f"{model_name}_summary.txt", 'w') as f:
            # Saves the code for the model layers
            f.write(model)
            
            # Saves size of train and validation set            
            f.write(f"Train_size: {training_lenght}\n")
            f.write(f"Validation_size: {validation_lenght}\n")

    # Creates model directory and returns the path.
    @classmethod
    def generate_model_path(cls, model_name):
        path = os.path.join(os.getcwd(), "models", model_name) + '/'
        cls.create_model_folder(path)
        return path
