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

        self.epoch_size = 2 # total number of runs
        self.batch_size = 128 # parts to split dataset into
        self.dataset_split_percentage = 0.9
        self.number_of_classes = 2
        self.TRAIN_PATH = 'data/train'
        self.VALIDATION_PATH = 'data/validation'
        self.training_lenght = 0
        self.validation_lenght = 0
        self.img_width, self.img_height = 128, 128

    def __create_model(self):
        '''
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
        model.add(Dense(2))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

        return model  
        '''

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten(name="intermediate_output"))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        # The extra metric is important for the evaluate function
        model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model

    def __fit_model_numpy(self, model, callbacks=None):
        animals_train, labels_train, animals_validation, labels_validation = self.__retrieve_dataset()
        history = model.fit(animals_train, labels_train, batch_size=self.batch_size, epochs=self.epoch_size, verbose=1, validation_data=(animals_validation, labels_validation), callbacks=callbacks)

        score = self.eval.evaluate_model(model)
        return history, score

    def __retrieve_dataset(self):
        animals_train, labels_train, animals_validation, labels_validation = self.ip.retrieve_train_validation(procent_split=self.dataset_split_percentage)

        self.training_lenght = len(animals_train)
        self.validation_lenght = len(animals_validation)

        return animals_train, labels_train, animals_validation, labels_validation

    def SVMWrapper(self, model):
        return ModelSVMWrapper(model)

    def TrainSVM(self, wrapper, callbacks=None):
        animals_train, labels_train, animals_validation, labels_validation = self.__retrieve_dataset()
        history = wrapper.fit(animals_train, labels_train, batch_size=self.batch_size, epochs=self.epoch_size, verbose=1, validation_data=(animals_validation, labels_validation), callbacks=callbacks)

        score = self.eval.evaluate_model(wrapper.model)
        return history, score

    def train_model(self):
        model = self.__create_model()

        #TensorBoard
        tensorboard_name = str(time())
        tensorboard = TensorBoard(log_dir="logs/{}".format(tensorboard_name));
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)
        
        history, score = self.__fit_model_numpy(model, [tensorboard, earlyStop])
        # history, score = self.TrainSVM(self.SVMWrapper(model), [tensorboard, earlyStop]) 

        #plot_history([('model', history)])
        score_rounded = f"{round(score[1], 3):.3f}"
        model_name = FileSystem.get_model_name(score_rounded, self.epoch_size, self.batch_size)
        model_path = FileSystem.get_model_path(model_name)

        print("[LOG] New model saved in " + model_path)
        model.save(model_path + model_name + ".h5")
        Graphs().plot_model(history, model_path, model_name)

        # Saves model
        lines = inspect.getsource(self.__create_model)
        FileSystem.save_model_summary(lines, model_path, model_name, self.training_lenght, 
                                        self.validation_lenght)

        # Renames model log
        FileSystem.rename_model_log(tensorboard_name, model_name)
        


if __name__ == "__main__":
    tm = TrainModel()
    tm.train_model()
