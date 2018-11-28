import keras
import numpy as np
from test import FoodDispenser, DispenseAgent

# Setup.
env = FoodDispenser()
env.predict = 1
state_size = env.state_size
action_size = env.action_size

agent = DispenseAgent(state_size, action_size) 
agent.load("food_dispenser.h5")


for x in range(0, 2):
    state = [env.predict, 0, 0]
    state = np.reshape(state, [1, state_size])
    result = agent.predict(state)
    next_state, reward, done = env.step(result)
    print(next_state)