import nxt
import numpy
import os
import nxt.locator

from nxt.sensor import *
from nxt.motor import *
from data.picture import take_pictures_CV2
from data.predicter import predict_folder
from time import sleep

ULTRASONICPORT = PORT_1
DIRECTORY = os.path.join(os.getcwd(), "pictures")
BOWLPOS = 0	 # 0 = in, 1 = out
BOWLROT = 0	 # 0 = cat, 1 = dog
BOWLPORT = PORT_B
MOTORPORT = PORT_A
GATEPORT = PORT_C
GATEPOS = 0	 # 0 = closed, 1 = open
MOTORSPEED = 0.5

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
			wait_and_close()
			
def print_console(input):
	print("[INFO] {}".format(input))

def dispense_food(animal):
	if BOWLROT != animal:
		rotate_bowl()
		
	# dispense food
	open_containers()
	# give cat food
	turn_gate()
	change_bowl_pos()
	
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
		
		if count >= 3:
			done = 1

	change_bowl_pos()
	turn_gate()

def open_containers():
	print_console("Opening food container")
	pass

def rotate_bowl():
	print_console("Rotating food bowl")
	# Rotate the bowl 180
	turn_motor(BOWLPORT, -45, 350)

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

if __name__ == "__main__":
	main()
