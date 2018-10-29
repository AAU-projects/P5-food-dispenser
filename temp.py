import os
import time

temp = 0
high = 0
low = "h"


def clear():
	_ = os.system('clear')


def measure_temp():
	global temp, high, low

	temp = os.popen("vcgencmd measure_temp").readline()
	temp = temp.replace("temp=","")
	if temp < low:
		low = temp

	if temp > high:
		high = temp
while True:
	
	measure_temp()
	print "Current temp: " +str(temp) + "\nHighest: " +str(high) + "\nLowest: " +str(low)
	time.sleep(2)
	clear()
