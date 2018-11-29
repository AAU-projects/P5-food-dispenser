import nxt.locator
from nxt.motor import *
from nxt.sensor import *

MOTORSPEED = 35
STEPSIZE = 90
BRICK = nxt.locator.find_one_brick()

def control_loop():
    while(True):
        text = input("cmds: opengate, closegate, bowlin, bowlout, rotbowlf, rotbowlt, portcharT/B:\n")
        
        if(text == "opengate"):
            turn_gate(False)
        elif(text == "closegate"):
            turn_gate(True)
        elif(text == "bowlin"):
            change_bowl_pos(False)
        elif(text == "bowlout"):
            change_bowl_pos(True)
        elif(text == "rotbowlf"):
            rotate_bowl(False)
        elif(text == "rotbowlt"):
            rotate_bowl(True)
        elif(text[0] == 'A'):
            step = int(text[3:])
            if(text[1] == 'F'):
                rotate_motor2(PORT_A, 1, step)
            elif(text[1] == 'T'):
                rotate_motor2(PORT_A, -1, step)
        elif(text[0] == 'B'):
            step = int(text[3:])
            if(text[1] == 'F'):
                rotate_motor2(PORT_B, 1, step)
            elif(text[1] == 'T'):
                rotate_motor2(PORT_B, -1, step)
        elif(text[0] == 'C'):
            step = int(text[3:])
            if(text[1] == 'F'):
                rotate_motor2(PORT_C, 1, step)
            elif(text[1] == 'T'):
                rotate_motor2(PORT_C, -1, step)
        else:
            print("Unrecognized input")
                
def turn_gate(reverse):
	if(reverse == True):
		rotate_motor2(PORT_C, -1, 630)
	else:
		rotate_motor2(PORT_C, 1, 630)
                
def change_bowl_pos(reverse):
	#  Drive bowl in and out
	if(reverse == True):
		rotate_motor2(PORT_A, 1, 350)
	else:
		rotate_motor2(PORT_A, -1, 350)

def rotate_bowl(reverse):
	# Rotate the bowl 180
  if(reverse == True):
	  rotate_motor2(PORT_B, -1, 250)
  else:
    rotate_motor2(PORT_B, 1, 250)

def rotate_motor2(port, speed, amount):
	motor = Motor(BRICK, port)
	motor.turn(int(speed * MOTORSPEED), amount)

def rotate_motor(port, speed):
	motor = Motor(BRICK, port)
	motor.turn(int(speed * MOTORSPEED), STEPSIZE)

if __name__ == '__main__':
    control_loop()
    
