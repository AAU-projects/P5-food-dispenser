import nxt
import numpy
import os

from nxt.sensor import *
from picture import take_pictures
from Model.PredictModel import predict


ULTRASONICPORT = PORT_1
DIRECTORY = os.path.join(os.getcwd(),"pictures")
BOWLPOS = 0 # 0 = in, 1 = out
BOWLROT = 0 # 0 = cat, 1 = dog

def main():
	controller = nxt.locator.find_one_brick()

	oldDist = Ultrasonic(controller, ULTRASONICPORT).get_sample()

	running = True
	while running:
		newDist = Ultrasonic(controller, ULTRASONICPORT).get_sample()
		
		if newDist != oldDist:
			oldDist = newDist
			take_pictures(DIRECTORY)

			# send pictures to ML
			result = predict(DIRECTORY)
			# do something based on ML respond
      			# Bowl is in
      			if BOWLPOS == 0:
        			# Its a dogo
				if result == 1:
          				# The bowl is in cat rotarion
          				if BOWLROT == 0:
						rotate_bowl()
          				# dispense food
          				dispense_food()
          				# give dogo food
          				change_bowl_pos()
        			# Its a cat
        			if result == 0:
          				# The bowl is in dog rotarion
          				if BOWLROT == 1:
				 		rotate_bowl()
          				# dispense food
          				dispense_food()
          				# give cat food
          				change_bowl_pos()
			else:
				change_bowl_pos()


def dispense_food():
	# Open for food
	pass

def rotate_bowl():
	# Rotate the bowl 180
	pass

def change_bowl_pos():
	# drive bowl in and out
	pass

def setup():
	if not os.path.exists(DIRECTORY):
    		os.makedirs(DIRECTORY)


if __name__ == "__main__":
	setup()
	main()
