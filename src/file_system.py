import os
from contextlib import redirect_stdout

class FileSystem:
    
    @classmethod
    def create_model_folder(cls, path):
        if not os.path.exists("models"):
            os.makedirs("models")

        os.makedirs(path)

    @classmethod
    def rename_model_log(cls, tensorboard_name, model_name):
        logpath = os.path.join(os.getcwd(),'logs', tensorboard_name)
        newlogpath = os.path.join(os.getcwd(),'logs', model_name)

        if os.path.exists(logpath):
            os.rename(logpath, newlogpath)

    @classmethod        
    def get_model_name(cls, score, epoch_size, batch_size):
        save_value = 1

        model_name = "model_{}_{}_{}".format(str(score), epoch_size, batch_size)
        new_model_name = model_name

        while(os.path.exists("models/" + new_model_name)):
            new_model_name = model_name + "({})".format(save_value)
            save_value += 1
            
        return new_model_name

    @classmethod
    def save_model_summary(cls, model, model_path, model_name, 
                            activation_functions, loss, optimizer, 
                            metrics, training_lenght, validation_lenght):
        # Save summary.
        with open(model_path + f"{model_name}_summary.txt", 'w') as f:
            with redirect_stdout(f):
                model.summary()
                
            for x in range(0, len(activation_functions)):
                f.write(f"activation_{x + 1} = {activation_functions[x]}\n")

            f.write(f"\nOptimizer: {optimizer}\n")    
            f.write(f"Loss: {loss}\n")    
            f.write(f"Metrics: {metrics}\n\n")    

            f.write(f"Train_size: {training_lenght}\n")
            f.write(f"Validation_size: {validation_lenght}\n")

    # Creates model directory and returns the path.
    @classmethod
    def get_model_path(cls, model_name):
        path = os.path.join(os.getcwd(), "models", model_name) + '/'
        cls.create_model_folder(path)
        return path
