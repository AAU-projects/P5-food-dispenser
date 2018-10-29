import time
import os
from picture import take_pictures_CV2 as take_pictures
from predicter import predict_folder

DIRECTORY = os.path.join(os.getcwd(), "pictures")

#Test

while True:
    #Take pictures
    take_pictures(DIRECTORY)
    print("pictures taken")

    #Predict pictures
    result = predict_folder(DIRECTORY)
    print(result)
    time.sleep(1)