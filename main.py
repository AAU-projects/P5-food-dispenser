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

CAT = 0
DOG = 1

BRICK = nxt.locator.find_one_brick()


def main():
	oldDist = Ultrasonic(BRICK, ULTRASONICPORT).get_sample()
	running = True
	print("Starting while loop")
	while running:
		print("Inside while loop")
		newDist = Ultrasonic(BRICK, ULTRASONICPORT).get_sample()

		if newDist != oldDist:
			oldDist = newDist
			
			take_pictures_CV2(DIRECTORY)
			print("Pics taken")

			# send pictures to ML
			result = predict_folder(DIRECTORY)
			
			# do something based on ML respond
			dispense_food(result)
			
			#Wait for the animal to finish, then close
			#wait_and_close()
			time.sleep(2)
			change_bowl_pos()
			turn_gate()
			

def dispense_food(animal):
	if BOWLROT != animal:
		rotate_bowl()
		
	# dispense food
	open_containers(animal)
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
		oldDist = Ultrasonic(BRICK, ULTRASONICPORT).get_sample()
		sleep(2)
		newDist = Ultrasonic(BRICK, ULTRASONICPORT).get_sample()
		
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
	global BOWLPOS
	#  Drive bowl in and out
	if BOWLPOS == 0:
		turn_motor(MOTORPORT, 45, 350)
		BOWLPOS = 1
	elif BOWLPOS == 1:
		turn_motor(MOTORPORT, -45, 350)
		BOWLPOS = 0


def turn_gate():
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

if __name__ == "__main__":
	main()
