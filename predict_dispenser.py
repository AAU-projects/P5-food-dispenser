import keras
import numpy as np
from test import FoodDispenser, DispenseAgent

NEXT_STATE = 1
BOWLROT = 0


def get_action(animal, agent):
    state = np.reshape(animal, [1, 1])
    action = agent.predict(state)
    return action

def rl_dispense_food(animal):
    # Setup enviroment

    # Setup agent
    agent = DispenseAgent(1, 2)
    agent.load("food_dispenser.h5")

    action = get_action(animal, agent)
    enviroment_step(action)

    # dispense food
    # open_containers(animal)


def enviroment_step(action):
    global NEXT_STATE

    if NEXT_STATE == 0:
        rotate_bowl_rl(action)
    elif NEXT_STATE == 1:
        rotate_dispenser_rl(action)
        NEXT_STATE = 0


def rotate_bowl():
    global BOWLROT
    BOWLROT = not BOWLROT


def rotate_bowl_rl(action):
    # No rotate bowl
    if (action == 0):
        print('not rotating bowl')
    # Rotate bowl
    elif (action == 1):
        print('rotating bowl')
        rotate_bowl()
    # if BOWLROT != action:

    # Increase function step
    global NEXT_STATE
    NEXT_STATE += 1

def rotate_dispenser_rl(action):
    # Dispense cat food
    if (action == 0):
        print('dispesning cat food')
        # If cat reward else not
    # Dispense dog food
    elif (action == 1):
        print('dispesning dog food')
        # If dog reward else not


if __name__ == "__main__":
    rl_dispense_food(1)
