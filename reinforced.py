from keras import Sequential
from keras.layers import Dense
from collections import deque
import random


# https://keon.io/deep-q-learning/
# http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
class DispenseAgent:
    def __init__(self):
        self.model = self._buildmodel()
        # Adds rewards in this array. 0 is rewards for the feed cat action, 1 is rewards for the feed dog action.
        self.rewards = []
        self.state_size = 1
        self.memory = deque(maxlen=2000)

    def _buildmodel(self):
        model = Sequential()
        model.add(Dense(4, input_dim=self.state_size, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        return model

    def act(self):
        pass

    def remember(self):
        self.memory.append("æææææh")
        pass

    def restart(self):
        pass

if __name__ == "__main__":
    pass