import os
import PIL
import glob
import cv2
import shutil
from random import choice
from time import sleep

WEBCAM = 0


def delete_all_files(destination):
	files = glob.glob(destination + "/*")
	for file in files:
		os.remove(file)


def take_pictures_CV2(destination, number_of_images=1):
	delete_all_files(destination)
	while True:
		try:
			capture = cv2.VideoCapture(WEBCAM)
			for x in range(0, number_of_images):
				ret, frame = capture.read()
				cv2.imwrite(os.path.join(destination, str(x)+".png"), frame)
				sleep(1)
			capture.release()
			break
		except Exception:
			print("[ERROR] image capture fail")
			capture.release()

def take_pictures_digital(destination, cats_dogs_destination):
	delete_all_files(destination)
	files = glob.glob(cats_dogs_destination + "/*")
	image = choice(files)
	print(len(files))
	print(image)
	image_destination = destination + "/0.jpg"
	print(image_destination)

	shutil.copy2(image, image_destination)

if __name__ == "__main__":
	take_pictures_CV2()


