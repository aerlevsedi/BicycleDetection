Bicycle Detection
=======

Based on
-----------
https://www.youtube.com/watch?v=bUoWTPaKUi4&ab_channel=Pysource

Imports
-----------
pip install opencv-python  
pip install matplotlip  
pip install numpy  

How to choose video
----------
Choose in code (lines 27-30) one of these and comment others:

Initialize camera  
cap = cv2.VideoCapture("highway.mp4")  
cap = cv2.VideoCapture("IMG_6048.MOV")  
cap = cv2.VideoCapture(0) # this takes view from the camera  
