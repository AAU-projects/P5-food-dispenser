import cv2
import os
from time import sleep

WEBCAM = 0

def take_pictures(destination = os.path.join(os.getcwd(),"pictures")):

	capture = cv2.VideoCapture(WEBCAM)
	for x in range(0, 10):
		ret, frame = capture.read()
		cv2.imwrite(os.path.join(destination, str(x)+".png"),frame)
		sleep(1)
	capture.release()



if __name__ == "__main__":
	take_pictures()
