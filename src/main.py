import cv2
import numpy as np
from gui_buttons import Buttons

buttons = Buttons()


colors = buttons.colors

# Opencv DNN
net = cv2.dnn.readNet("C:\\Users\\HP\\Desktop\\STUDIA\\BicycleDetection-main\\src\\dnn_model\\yolov4-tiny.weights",
                      "C:\\Users\\HP\\Desktop\\STUDIA\\BicycleDetection-main\\src\\dnn_model\\yolov4-tiny.cfg")
# net = cv2.dnn.readNet("C:\\Users\\HP\\Desktop\\STUDIA\\BicycleDetection-main\\src\\dnn_model\\yolov4-tiny.weights", "C:\\Users\\HP\\Desktop\\STUDIA\\BicycleDetection-main\\src\\dnn_model\\yolov3-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(640, 640), scale=1/255)

# Load class list
classes = []
with open("C:\\Users\\HP\\Desktop\\STUDIA\\BicycleDetection-main\\src\\dnn_model\\classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


# Initialize camera
# cap = cv2.VideoCapture("highway.mp4")
cap = cv2.VideoCapture(
    "C:\\Users\\HP\\Desktop\\STUDIA\\BicycleDetection-main\\src\\IMG_6048_new.mp4")
# cap = cv2.VideoCapture(0) # this takes view from the camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 850)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Full HD 1920 x 1080


def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        buttons.button_click(x, y)


# Create window
cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Frame", click_button)

counter = 0

while True:
    # Get frames
    ret, frame = cap.read()

    # Object detection
    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox

        class_name = classes[class_id]
        color = colors[class_id]

        if class_name == "bicycle":
            print(score, ";", counter)
            cv2.putText(frame, str(score), (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    cv2.imshow("Frame", frame)
    counter += 1
    key = cv2.waitKey(1)
    if key == 27:  # Press Escape to exit
        break

cap.release()
cv2.destroyAllWindows()
