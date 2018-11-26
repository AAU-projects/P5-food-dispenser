import keras
import test
import numpy as np

#Setup
env = test.FoodDispenser()
agent = test.DispenseAgent(3, 2)
agent.load("food_dispenser.h5")

env = test.FoodDispenser()
state_size = env.state_size
action_size = env.action_size
agent = test.DispenseAgent(state_size, action_size)


# Predict

# 0 for cat
# 1 for dog
state = env.predict_functions(0)

state = np.reshape(state, [1, state_size])
for step in range(2):
    action = agent.act(state)
    next_state, reward, done = env.step(action)

    reward = reward if done else -1
    next_state = np.reshape(next_state, [1, state_size])
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    print(state)
            