from keras import Sequential
from keras.layers import Dense


# https://keon.io/deep-q-learning/
# http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
class DispenseAgent:
    def __init__(self):
        self.model = self._buildmodel()

    def _buildmodel(self):
        model = Sequential()
        model.add(Dense(4, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        return model

    def act(self):
        pass