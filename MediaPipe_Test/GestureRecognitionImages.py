import cv2 
import mediapipe as mp
import math

FILENAMES = ["MediaPipe_Test/images/thumbs_down.jpg", "MediaPipe_Test/images/thumbs_up.jpg", "MediaPipe_Test/images/finger_up.jpg", "MediaPipe_Test/images/peace.jpg"]
HEIGHT, WIDTH = 480, 480

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (WIDTH, math.floor(HEIGHT * h / w)))
    else:
        img = cv2.resize(image, (math.floor(w/(h/HEIGHT)), HEIGHT))
    cv2.imshow("Image",img)
    cv2.waitKey(0)

def run_resize_and_show():
    images = {name: cv2.imread(name) for name in FILENAMES}
    for name, image in images.items():
        print(name, image.shape)
        resize_and_show(image)