import inspect
from keras_svm import ModelSVMWrapper
from src.graphs import Graphs
from src.image_processing import ImageProcessing
from src.file_system import FileSystem
from src.model_evaluate import ModelEvaluate
from time import time
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping

class TrainModel:
    def __init__(self, *args, **kwargs):
        self.ip = ImageProcessing()
        self.eval = ModelEvaluate()

        self.epoch_size = 10   # total number of runs
        self.batch_size = 128 # parts to split dataset into
        self.dataset_split_percentage = 0.9
        self.number_of_classes = 3
        self.training_lenght = 0
        self.validation_lenght = 0
        self.img_width, self.img_height = 128, 128

    def __create_model(self):
        
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(self.img_width, self.img_height, 3)))
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

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.number_of_classes))
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
        model = self.__create_model()

        # Tensor Callbacks, TensorBoard and Early stopping
        tensorboard_name = str(time())
        tensorboard = TensorBoard(log_dir="logs/{}".format(tensorboard_name));
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)
        
        # Train the model
        history, score = self.__fit_model(model, [tensorboard, earlyStop])

        # Generate the name of the just trained model, and get the file location to store it at
        score_rounded = f"{round(score[1], 3):.3f}"
        model_name = FileSystem.get_model_name(score_rounded, self.epoch_size, self.batch_size)
        model_path = FileSystem.get_model_path(model_name)

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
        FileSystem.rename_model_log(tensorboard_name, model_name)
        

if __name__ == "__main__":
    tm = TrainModel()
    tm.train_model()
