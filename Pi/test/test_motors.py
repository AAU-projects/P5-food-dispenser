import nxt.locator
from nxt.motor import *
from nxt.sensor import *

import time
import sys

def spin_around(b, lr):
    speed = 15 * 3
    range = 350
    ready_state = False
    
    while(not ready_state):
        sonicValue = Ultrasonic(b, PORT_1).get_sample()
        print(sonicValue)
        #When range is 20 move bowl forward
        if (sonicValue <= 20):
            m_rigth = Motor(b, PORT_C)
            m_rigth.turn(speed, range * 1.8)
            ready_state = True
            #Rotate bowl
            m_rigth = Motor(b, PORT_B)
            m_rigth.turn(-speed, range)
            #Move bowl forward
            m_left = Motor(b, PORT_A)
            m_left.turn(speed, range)
        time.sleep(0.1)

    time.sleep(2)
    #Move bowl back
    m_rigth = Motor(b, PORT_A)
    m_rigth.turn(-speed, range)

    #CLOSE GATE
    m_rigth = Motor(b, PORT_C)
    m_rigth.turn(-speed, range * 1.8)

    

def spin_motor(b, lr):
    speed = 10 
    range = 350
    if (lr == 'a'):
        m_left = Motor(b, PORT_A)
        m_left.turn(speed, range)
    if (lr == 'b'):
        m_left = Motor(b, PORT_B)
        m_left.turn(speed, range)

b = nxt.locator.find_one_brick()

spin_around(b, sys.argv[1])