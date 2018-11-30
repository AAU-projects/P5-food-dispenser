import nxt
import numpy
import os
import nxt.locator
import random
import numpy as np

from nxt.sensor import *
from nxt.motor import *
from data.picture import take_pictures_CV2
from data.predicter import predict_folder
from time import sleep
from rl_agent_env import DispenseAgent, FoodDispenser

ULTRASONICPORT = PORT_1
DIRECTORY = os.path.join(os.getcwd(), "pictures")
BOWLPOS = 0	 # 0 = in, 1 = out
BOWLROT = 0	 # 0 = cat, 1 = dog
BOWLPORT = PORT_B
MOTORPORT = PORT_A
GATEPORT = PORT_C
GATEPOS = 0	 # 0 = closed, 1 = open
MOTORSPEED = 0.5
NEXT_STATE = 0

CAT = 0
DOG = 1

BRICK = nxt.locator.find_one_brick()

def main():
	oldDist = get_range()
	running = True
	while running:
		print_console("Waiting for movement...")
		newDist = get_range()

		if newDist != oldDist:
			oldDist = newDist
			
			print_console("Taking pictures")
			take_pictures_CV2(DIRECTORY)

			# Pictures analyzed ML
			print_console("Predicting from pictures")
			result = predict_folder(DIRECTORY)
			
			# do something based on ML respond
			dispense_food(result)
			
			#Wait for the animal to finish, then close
			#wait_and_close()
			time.sleep(2)
			change_bowl_pos()
			turn_gate()
			
def print_console(input):
	print("[INFO] {}".format(input))

def dispense_food(animal):

	rl_dispense_food(animal)
		

	# Open gate
	turn_gate()
  # Push bowl out
	change_bowl_pos()

def bowl_forward_small():
  turn_motor(MOTORPORT, 45, 20)
  
def bowl_backward_small():
  turn_motor(MOTORPORT, -45, 20)
	
def wait_and_close():
	done = 0
	count = 0
	
	while(not done):
		oldDist = get_range()
		sleep(2)
		newDist = get_range()
		
		if oldDist == newDist:
			count += 1
			print("WaitCount: " + str(count))
		else:
			count = 0
		
		if count >= 3:
			done = 1

	change_bowl_pos()
	turn_gate()

def open_containers(animal):
  bowl_forward_small()
  if(animal == CAT):
    turn_motor(BOWLPORT, 45, 90)
    sleep(1)
    turn_motor(BOWLPORT, -45, 90)
  else:
    turn_motor(BOWLPORT, -45, 90)
    sleep(1)
    turn_moter(BOWLPORT, 45, 90)
  bowl_backward_small()

def rotate_bowl():
	# Rotate the bowl 180 by moving it back and forward to make sure we dont dispense the food prematurely.
  turn_motor(BOWLPORT, 45, 50)
  bowl_forward_small()
  turn_motor(BOWLPORT, -45, 50)
  bowl_backward_small()
  turn_motor(BOWLPORT, 45, 50)
  bowl_forward_small()
  turn_motor(BOWLPORT, -45, 50)

def change_bowl_pos():
	print_console("Moving bowl")
	global BOWLPOS
	#  Drive bowl in and out
	if BOWLPOS == 0:
		turn_motor(MOTORPORT, 45, 350)
		BOWLPOS = 1
	elif BOWLPOS == 1:
		turn_motor(MOTORPORT, -45, 350)
		BOWLPOS = 0

def turn_gate():
	print_console("Turning gate")
	global GATEPOS
	if GATEPOS == 0:
		turn_motor(GATEPORT, 45, 630)
		GATEPOS = 1
	elif GATEPOS == 1:
		turn_motor(GATEPORT, -45, 630)
		GATEPOS = 0

def turn_motor(port, speed, range):
	motor = Motor(BRICK, port)
	motor.turn(int(speed * MOTORSPEED), range)

def get_range():
	return Ultrasonic(BRICK, ULTRASONICPORT).get_sample()

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
	main()
