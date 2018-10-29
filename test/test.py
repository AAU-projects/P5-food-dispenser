import os

from picture import take_pictures_digital as take_picture
from predicter import predict_folder

DIRECTORY = os.path.join(os.getcwd(),"pictures") 
DIRECTORY_CATS_DOGS = os.path.join(os.getcwd(),"random_dog_cat")


take_picture(DIRECTORY, DIRECTORY_CATS_DOGS)
result = predict_folder(DIRECTORY)
print(result)
