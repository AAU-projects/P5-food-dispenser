import matplotlib.pyplot as plt


class Graphs:
    def plot_model(self, history, model_path, model_name):

        plt.plot(history.epoch, history.history['acc'], '--', label="Training Accuracy", color='b')
        plt.plot(history.epoch, history.history['val_acc'], '-', label="Validation Accuracy", color='g')

        plt.plot(history.epoch, history.history['categorical_crossentropy'], '-.', label="Categorical Crossentropy", color='m')
        plt.plot(history.epoch, history.history['val_categorical_crossentropy'], '-.', label="Categorical Crossentropy", color='m')

        plt.plot(history.epoch, history.history['loss'], '--', label="Training Loss", color='b')
        plt.plot(history.epoch, history.history['val_loss'], '-', label="Validation Loss", color='g')

        plt.title(f'{model_name}')
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(model_path + "{}_graph.png".format(model_name))