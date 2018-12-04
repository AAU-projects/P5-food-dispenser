import os
import time

temp = 0.0
high = 0.0
low = 100.0


def clear():
	_ = os.system('clear')


def measure_temp():
	global temp, high, low

	temp = os.popen("vcgencmd measure_temp").readline()
	temp = temp.replace("temp=","")
	temp = temp.replace("'C\n", "")
	if float(temp) < float(low):
		low = temp

	if float(temp) > float(high):
		high = temp
while True:
	
	measure_temp()
	if float(temp) >= 85.0:
		os.system("sudo shutdown now -h")
	print("Current temp: " +str(temp) + "\nHighest: " +str(high) + "\nLowest: " +str(low))
	time.sleep(1)
	clear()
