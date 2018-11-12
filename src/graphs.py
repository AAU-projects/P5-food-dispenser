import matplotlib.pyplot as plt


class Graphs:
    def plot_model(self, history, model_path, model_name):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.plot(history.history['binary_crossentropy'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['val_binary_crossentropy'])

        plt.plot(history.epoch, history.history['acc'], '--', label="Training Accuracy",)
        plt.plot(history.epoch, history.history['val_acc'], '-', label="Validation Accuracy")
        
        plt.plot(history.epoch, history.history['binary_crossentropy'], '-.', label="Binary Crossentropy")
        plt.plot(history.epoch, history.history['binary_crossentropy'], '-.', label="Binary Crossentropy")

        plt.plot(history.epoch, history.history['acc'], '--', label="Training Loss", )
        plt.plot(history.epoch, history.history['val_acc'], '-', label="Validation Loss")

        plt.title(f'{model_name}')
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train_acc', 'Test_acc', "Train_loss", "Test_loss"], loc='upper left')
        plt.savefig(model_path + "{}_graph.png".format(model_name))
