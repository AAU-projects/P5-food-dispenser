import nxt.locator
from nxt.motor import *
from nxt.sensor import *

MOTORSPEED = 22   # found from trial and error
STEPSIZE = 90
BRICK = nxt.locator.find_one_brick()


def control_loop():
    while(True):
        text = input("cmds: opengate, closegate, bowlin, bowlout, rotbowl+, rotbowl-, portchar+/-:\n")
        
        if(text == "opengate"):
            turn_gate(False)
        elif(text == "closegate"):
            turn_gate(True)
        elif(text == "bowlin"):
            change_bowl_pos(False)
        elif(text == "bowlout"):
            change_bowl_pos(True)
        elif(text == "rotbowl+"):
            rotate_bowl(False)
        elif(text == "rotbowl-"):
            rotate_bowl(True)
        elif(text[0] == 'A'): # MOTORPORT = PORT A
            step = int(text[3:])
            if(text[1] == '+'):
                rotate_motor(PORT_A, 1, step)
            elif(text[1] == '-'):
                rotate_motor(PORT_A, -1, step)
        elif(text[0] == 'B'): # BOWLPORT = PORT B
            step = int(text[3:])
            if(text[1] == '+'):
                rotate_motor(PORT_B, 1, step)
            elif(text[1] == '-'):
                rotate_motor(PORT_B, -1, step)
        elif(text[0] == 'C'): # GATEPORT = PORT C
            step = int(text[3:])
            if(text[1] == '+'):
                rotate_motor(PORT_C, 1, step)
            elif(text[1] == '-'):
                rotate_motor(PORT_C, -1, step)
        else:
            print("Unrecognized input")
                
def turn_gate(reverse):
    # Opens the gate
	if(reverse == True):
		rotate_motor(PORT_C, -1, 630) # rotate_motor(port, speed, degrees)
	else:
		rotate_motor(PORT_C, 1, 630)

def change_bowl_pos(reverse):
	# Drive bowl in and out
	if(reverse == True):
		rotate_motor(PORT_A, 1, 350)
	else:
		rotate_motor(PORT_A, -1, 350)

def rotate_bowl(reverse):
    # Rotate the bowl 180
    if(reverse == True):
        rotate_motor(PORT_B, -1, 250)
    else:
        rotate_motor(PORT_B, 1, 250)

def rotate_motor(port, speed, amount):
	motor = Motor(BRICK, port)
	motor.turn(int(speed * MOTORSPEED), amount)

if __name__ == '__main__':
    control_loop()
    