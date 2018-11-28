# -*- coding: utf-8 -*-
import random
import gym
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
        self.state_size = state_size # input size
        self.action_size = action_size # output size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 # min exploration rate
        self.epsilon_decay = 0.995 
        
        # Adds rewards in this array. 0 is rewards for the feed cat action, 1 is rewards for the feed dog action.
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

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def predict(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class FoodDispenser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predict = random.randint(0, 1)
        self.reward = 0
        self.next_state = 0
        self.state_size = 3 # input size
        self.action_size = 2 # output size

        #Food dispenser (0 cat, 1 dog)
        self.bowl_rotation = 0
        self.dispense_rotation = 0

        self.done = False

        return [self.predict, self.bowl_rotation, self.dispense_rotation]

    def rotate_bowl(self, action):
        # No rotate bowl
        if (action == 0):
            self.bowl_rotation = 0
            # If cat reward else not            
            if self.predict == 0:
                self.reward += 1
            else:
                self.reward += -1  
                
        # Rotate bowl          
        elif (action == 1):
            self.bowl_rotation = 1
            # If dog reward else not
            if self.predict == 1:
                self.reward += 1
            else:
                self.reward += -1

        # Increase function step
        self.next_state += 1

    def rotate_dispenser(self, action):
        # Dispense cat food
        if (action == 0):
            self.dispense_rotation = 0            
            # If cat reward else not
            if self.predict == 0:
                self.reward += 1
            else:
                self.reward += -1
        # Dispense dog food
        elif (action == 1):
            self.dispense_rotation = 1
            # If dog reward else not
            if self.predict == 1:
                self.reward += 1
            else:
                self.reward += -1 

    def step(self, action):
        if self.next_state == 0:
            self.rotate_bowl(action)
        elif self.next_state == 1:
            self.rotate_dispenser(action)
            self.done = True
            
        return [self.predict, self.bowl_rotation, self.dispense_rotation], self.reward, self.done

EPISODES = 1000
if __name__ == "__main__":
    env = FoodDispenser()
    state_size = env.state_size
    action_size = env.action_size
    agent = DispenseAgent(state_size, action_size)
    #agent.load("food_dispenser.h5")
    done = False
    batch_size = 32

    for e in range(0, EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for step in range(2):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            #reward = reward if done else -1

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Done becomes True when no more steps 
            # The agent drops the pole
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, step {}"
                        .format(e + 1, EPISODES, reward, agent.epsilon, step))
                break
            # train the agent with the experience of the episode
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
    print('Saving model')
    agent.save("food_dispenser.h5")   





































































































































































