# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''
INPUT:
    [0] - Model predicted that the image was of a cat
    [1] - Model predicted that the image was of a dog

OUTPUT:
    [0] - rotate for cat
    [1] - rotate for dog
'''
# https://keon.io/deep-q-learning/
# http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/


class DispenseAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size                    # Input size.
        self.action_size = action_size                  # Output size.
        self.gamma = 0.95                               # Discount rate.
        self.epsilon = 1.0                              # Exploration rate.
        self.epsilon_min = 0.01                         # Min exploration rate.
        self.epsilon_decay = 0.95

        self.memory = deque(maxlen=2000)

        self.model = self._buildmodel()

    def _buildmodel(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam')
        return model

    # Will either guess what action to take, or try to use the model to predict an action.
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)   # Guesses and action.
        act_values = self.model.predict(state)          # Predicts an action.
        return np.argmax(act_values[0])                 # Returns action.

    # Predicts an action
    def predict(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])                 # Returns action.

    # Remembers the performed action in a state
    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

    # Used to train the model with a sample(minibatch) of previous memories.                
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)              # Create a minibatch from the memory, with the size of {batch_size}.
        for state, action, reward, next_state, done in minibatch:       # Iterate the minibatch.
            target = reward                                             
            if not done:
                target = (reward + self.gamma *                         # Target for Q-Learning is found. This is used later by the fit() function.                                                
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:                             # If epsilon has not reached the minimum yet, decrease epsilon by decay amount.
            self.epsilon *= self.epsilon_decay

    # Loads model
    def load(self, name):
        self.model.load_weights(name)
    
    # Saves model    
    def save(self, name):
        self.model.save_weights(name)

class FoodDispenser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predict = random.randint(0, 1) # Random if cat or dog.
        self.reward = 0
        self.next_state = 0
        self.state_size = 1                 # Input size.
        self.action_size = 2                # Output size.

        self.bowl_position = 0
        self.dispenser_position = 0

        self.done = False

        return [self.predict, self.bowl_position, self.dispenser_position]

    # Determines what reward to give the agent, depending on the action taken.
    def rotate_bowl(self, action):
        # Rotate bowl for cat food.
        if action == 0:                     # Action is: don't rotate bowl.
            self.bowl_position = 0          # Set bowl_position to use with the next_state.
            if self.predict == 0:           # Action is the same as the prediction, therefore reward. (Cat food)
                self.reward += 1
            else:
                self.reward += -1           # Action is not the same as the prediction, therefore punish.

        # Rotate bowl for dog food.
        elif action == 1:                   # Action is: rotate bowl.
            self.bowl_position = 1          # Set bowl_position to use with the next_state.
            if self.predict == 1:           # Action is the same as the prediction, therefore reward. (Dog food)
                self.reward += 1
            else:
                self.reward += -1           # Action is not the same as the prediction, therefore punish.

        self.next_state += 1                # Increase function step.

    # Determines what reward to give the agent, depending on the action taken.
    def rotate_dispenser(self, action):
        # Dispense cat food.
        if (action == 0):
            self.dispenser_position = 0
            # If cat reward else not
            if self.predict == 0:
                self.reward += 1
            else:
                self.reward += -1
        # Dispense dog food.
        elif (action == 1):
            self.dispenser_position = 1
            # If dog reward else not
            if self.predict == 1:
                self.reward += 1
            else:
                self.reward += -1 

    # Determines what function to execute depending on the {next_state}. Returns the current state.
    def step(self, action):
        if self.next_state == 0:
            self.rotate_bowl(action)
        elif self.next_state == 1:
            self.rotate_dispenser(action)
            self.done = True
            
        return [self.predict, self.bowl_position, self.dispenser_position], self.reward, self.done

EPISODES = 200
if __name__ == "__main__":
    env = FoodDispenser()                                                           # Create new environment.
    agent = DispenseAgent(env.state_size, env.action_size)                          # Create new agent.
    done = False
    batch_size = 16                                                                 # Set batch size.

    for e in range(0, EPISODES):
        state = env.reset()                                                         # State set to predict variable of environment. (0 or 1)

        for step in range(2):                                                       # Go through all steps.
            action = agent.act(state)                                               # The agent makes an action from the current state.
            next_state, reward, done = env.step(action)                             # Takes an action, and returns the state of the environment.

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:                                                                # Done becomes True when no more steps.
                print("episode: {}/{}, score: {}, e: {:.2}, step {}"
                        .format(e + 1, EPISODES, reward, agent.epsilon, step))
                break

            if len(agent.memory) > batch_size:                                      # Train the agent with the experience of the episode.
                agent.replay(batch_size)
                
    print('Saving model')
    agent.save("food_dispenser.h5")   