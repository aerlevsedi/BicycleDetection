import cv2
import numpy as np
from gui_buttons import Buttons

buttons = Buttons()
buttons.add_button("person", 20, 20)
buttons.add_button("cup", 20, 80)
buttons.add_button("bicycle", 20, 140)
buttons.add_button("car", 20, 200)

colors = buttons.colors

# Opencv DNN
net = cv2.dnn.readNet("dnn_model\\yolov4-tiny.weights", "dnn_model\\yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class list
classes = []
with open("dnn_model\\classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print(classes)

# Initialize camera
# cap = cv2.VideoCapture("highway.mp4")
cap = cv2.VideoCapture("IMG_6048.MOV")
# cap = cv2.VideoCapture(0) # this takes view from the camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Full HD 1920 x 1080

button_person = False

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        buttons.button_click(x, y)



#Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    # Get frames
    ret, frame = cap.read()

    # Get active buttons list
    active_buttons = buttons.active_buttons_list()
    print("Active buttons", active_buttons)

    # Object detection
    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox

        class_name = classes[class_id]
        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Display buttons
    buttons.display_buttons(frame)
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) 
    if key == 27: # Press Escape to exit
        break

cap.release()
cv2.destroyAllWindows()