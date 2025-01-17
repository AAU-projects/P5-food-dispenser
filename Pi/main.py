import nxt
import numpy
import os
import nxt.locator
import random
import numpy as np
import warnings
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
JUNK = 2

BRICK = nxt.locator.find_one_brick()

# Setup agent
agent = DispenseAgent(1, 2)
agent.load("food_dispenser.h5")

def main():
	# Get distance from sensor
	oldDist = get_range()
	running = True
	print_console("Waiting for movement...")
	while running:
		# Get distance from sensor
		newDist = get_range()

		# If oldDist != newdist take picture
		if abs(newDist - oldDist) > 5:
			oldDist = newDist
			result = -1
			
			while True:
				try:
					print_console("Taking pictures")
					take_pictures_CV2(DIRECTORY)

					# Classify picture
					print_console("Predicting from pictures")
					result = predict_folder(DIRECTORY)
					print(result)
					break
				except Exception:
					print_console("Image capture fail", "ERROR")
					print_console("Trying again")
			
			if not (result == -1 or result == 2):
				# Send result to the reinforment model
				rl_dispense_food(result)
				
				# Wait for the animal to finish, then close
				wait_and_close()
				
				if result == CAT:
					# Return bowl to start position
					rotate_bowl()
			else:
				print_console("Junk")

# Prints message to console
def print_console(input, type_print="INFO"):
	print("[{0}] {1}".format(type_print, input))

# Moves the bowl a small distance forward
def bowl_forward_small():
  turn_motor(MOTORPORT, 45, 20)
  
# Moves the bowl a small distance backward  
def bowl_backward_small():
  turn_motor(MOTORPORT, -45, 20)

# Waits until there is no significant change in distance, 
# meaning the pet is not infront of the bowl. Then retracts the bowl
# and closes the gate	
def wait_and_close():
	done = 0
	count = 0
	wait_count = 3
	
	while(not done):
		oldDist = get_range()
		sleep(2)
		newDist = get_range()
		
		if oldDist == newDist:
			count += 1
			print_console("Wait count: {0}/{1}".format(count, wait_count))
		else:
			count = 0
			print_console("Movement detected")
		
		if count >= wait_count:
			done = 1

	change_bowl_pos()
	turn_gate()

# Open food containers according to the animal 
def open_containers(action):
	bowl_forward_small()
	# Dispense cat food
	if(action == CAT):
		turn_motor(BOWLPORT, 45, 90)
		sleep(1)
		turn_motor(BOWLPORT, -45, 90)
	# Dispense dog food 
	elif (action == DOG):
		turn_motor(BOWLPORT, -45, 90)
		sleep(1)
		turn_motor(BOWLPORT, 45, 90)
	bowl_backward_small()

def rotate_bowl():
	# Rotate the bowl 180 by moving it back and forward to make sure we dont dispense the food prematurely.
	turn_motor(BOWLPORT, 45, 35)
	bowl_forward_small()
	turn_motor(BOWLPORT, -45, 35)
	bowl_backward_small()
	turn_motor(BOWLPORT, 45, 36)
	bowl_forward_small()
	turn_motor(BOWLPORT, -45, 36)
	bowl_backward_small()

	global BOWLROT 
	BOWLROT = not BOWLROT

def change_bowl_pos():
	global BOWLPOS
	print_console("Moving bowl")
	#  Drive bowl in and out
	if BOWLPOS == 0:
		turn_motor(MOTORPORT, 45, 350)
	elif BOWLPOS == 1:
		turn_motor(MOTORPORT, -45, 350)

	BOWLPOS = not BOWLPOS

def turn_gate():
	global GATEPOS
	print_console("Turning gate")
	# If gate is closed then open
	if GATEPOS == 0:
		turn_motor(GATEPORT, 45, 630)
	# If date is open then close
	elif GATEPOS == 1:
		turn_motor(GATEPORT, -45, 630)
		
	GATEPOS = not GATEPOS

# Turns the motor connected to {port} on the NXT. A negative speed value turns the motor the opposite direction.
def turn_motor(port, speed, range):
	motor = Motor(BRICK, port)
	motor.turn(int(speed * MOTORSPEED), range)

# Returns the range value from the ultrasonic sensor connected to the NXT.
def get_range():
	return Ultrasonic(BRICK, ULTRASONICPORT).get_sample()

# Gets a action from the agent
def get_action(animal):
	action = agent.predict([animal])
	return action

# Using Reinforcement learning to dispense the food
def rl_dispense_food(animal):
	action = get_action(animal)
	enviroment_step(action)

	action = get_action(animal)
	enviroment_step(action)

	# Rotate the bowl so the right side faces outward to the animal
	rotate_bowl()
	
	# Open gate
	turn_gate()
	
	# Push bowl out
	change_bowl_pos()
 
# Depending on the enviroment state and action the bowl is rotated or the containers are opened.
def enviroment_step(action):
	global NEXT_STATE

	if NEXT_STATE == 0:
		# Bowl receives an action
		rotate_bowl_rl(action)
		NEXT_STATE += 1
	elif NEXT_STATE == 1:
		# Containers receive an action
		open_containers_rl(action)
		NEXT_STATE = 0

def rotate_bowl_rl(action):
	# No rotation of bowl
	if (action == 0):
		print_console('Not rotating bowl')
	# Rotate bowl
	elif (action == 1):
		print_console('Rotating bowl')
		rotate_bowl()

def open_containers_rl(action):
	# No rotation of container
	if (action == 0):
		print_console('Rotating container to cat')
		open_containers(CAT)
	# Rotate container
	elif (action == 1):
		print_console('Rotating container to dog')
		open_containers(DOG)

if __name__ == "__main__":
	main()
