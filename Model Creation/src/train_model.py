import inspect
import src.CommonVars as vars
from src.graphs import Graphs
from src.image_processing import ImageProcessing
from src.file_system import FileSystem
from src.model_evaluate import ModelEvaluate
from time import time
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization

class TrainModel:
    def __init__(self, *args, **kwargs):
        self.ip = ImageProcessing()
        self.eval = ModelEvaluate()
        self.training_lenght = 0
        self.validation_lenght = 0

        # Variables for the training
        self.epoch_size = 100                   # Total number of runs
        self.batch_size = 128                   # Parts to split dataset into
        self.dataset_split_percentage = 0.9     # The split of the dataset for training and validation


    def __create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(vars.img_height, vars.img_width, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Conv2D(48, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(48, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Flatten()) 
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(vars.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

        return model

    def __fit_model(self, model, callbacks=None):
        # Get images
        pictures_train, labels_train, pictures_validation, labels_validation = self.__retrieve_dataset()

        # Start the training
        history = model.fit(pictures_train, labels_train, batch_size=self.batch_size, epochs=self.epoch_size, verbose=1, validation_data=(pictures_validation, labels_validation), callbacks=callbacks)

        # Evaluate the model
        score = self.eval.evaluate_model(model)
        return history, score

    def __retrieve_dataset(self):
        pictures_train, labels_train, pictures_validation, labels_validation = self.ip.retrieve_train_validation(procent_split=self.dataset_split_percentage)

        # Number of training pictures
        self.training_lenght = len(pictures_train)
        # Number of validation pictures
        self.validation_lenght = len(pictures_validation)

        return pictures_train, labels_train, pictures_validation, labels_validation

    def train_model(self):
        # Creates the model
        model = self.__create_model()

        # Train the model
        history, score = self.__fit_model(model, vars.callbacks)

        # Generate the name of the just trained model, and get the file directory to store it at
        score_rounded = f"{round(score[1], 3):.3f}"
        model_name = FileSystem.generate_model_name(score_rounded, self.epoch_size, self.batch_size)
        model_path = FileSystem.generate_model_path(model_name)

        # Save the model to disk
        print("[LOG] New model saved in " + model_path)
        model.save(model_path + model_name + ".h5")

        # Create graphs of model performance
        Graphs().plot_model(history, model_path, model_name)

        # Save model summery
        lines = inspect.getsource(self.__create_model)
        FileSystem.save_model_summary(lines, model_path, model_name, self.training_lenght,
                                        self.validation_lenght)

        # Renames model log
        FileSystem.rename_model_log(vars.tensorboard_name, model_name)


if __name__ == "__main__":
    tm = TrainModel()
    tm.train_model()
