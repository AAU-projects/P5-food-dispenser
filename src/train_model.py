from graphs import Graphs
from image_processing import ImageProcessing
from file_system import FileSystem
from time import time
from model_evaluate import ModelEvaluate
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from keras.callbacks import TensorBoard, EarlyStopping

class TrainModel:
    def __init__(self, *args, **kwargs):
        self.epoch_size = 2 # total number of runs
        self.batch_size = 64 # parts to split dataset into
        self.TRAIN_PATH = 'data/train'
        self.VALIDATION_PATH = 'data/validation'

        self.training_lenght = 0
        self.validation_lenght = 0

        self.img_width, self.img_height = 128, 128

        self.activation_functions = ['relu', 'relu', 'relu', 'relu', 'softmax']
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy', 'categorical_crossentropy']

    def __create_model(self):    
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(self.img_width, self.img_height, 3)))
        model.add(Activation(self.activation_functions[0]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation(self.activation_functions[1]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(self.activation_functions[2]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation(self.activation_functions[3]))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation(self.activation_functions[4]))

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return model  

    def __fit_model_numpy(self, model, callbacks=None):
        animals_train, labels_train, animals_validation, labels_validation = self.__retrieve_dataset()
        history = model.fit(animals_train, labels_train, batch_size=self.batch_size, epochs=self.epoch_size, verbose=1, callbacks=callbacks)

        score = ModelEvaluate.evaluate_model(model)
        return history, score

    def __retrieve_dataset(self):
        ip = ImageProcessing()
        animals_train, labels_train, animals_validation, labels_validation = ip.retrieve_train_validation(procent_split=0.8)

        self.training_lenght = len(animals_train)
        self.validation_lenght = len(animals_validation)

        return animals_train, labels_train, animals_validation, labels_validation

    def train_model(self):
        model = self.__create_model()

        #TensorBoard
        tensorboard_name = str(time())
        tensorboard = TensorBoard(log_dir="logs/{}".format(tensorboard_name));
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)

        history, score = self.__fit_model_numpy(model, [tensorboard, earlyStop])

        #plot_history([('model', history)])
        score_rounded = f"{round(score[1], 3):.3f}"
   
        model_name = FileSystem.get_model_name(score_rounded, self.epoch_size, self.batch_size)
        model_path = FileSystem.get_model_path(model_name)

        print("[LOG] New model saved in " + model_path)
        model.save(model_path + model_name + ".h5")
        Graphs().plot_model(history, model_path, model_name)
        FileSystem.save_model_summary(model, model_path, model_name, self.activation_functions, 
                                        self.loss, self.optimizer, self.metrics, self.training_lenght, 
                                        self.validation_lenght)
        FileSystem.rename_model_log(tensorboard_name, model_name)


if __name__ == "__main__":
    TrainModel().train_model()
