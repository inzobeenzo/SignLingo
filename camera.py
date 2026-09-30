"""
THIS IS A TEST ENVIRONMENT NOT THE REAL SCRIPT
"""

import numpy as np
import mediapipe as mp
import cv2 as cv
import time
import math

python = mp.tasks
vision = mp.tasks.vision

latest_gesture = "No Gesture"
latest_hand_landmarks = None

def save_result(result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture, latest_hand_landmarks
    if result.gestures and len(result.gestures) > 0:
        top_gesture = result.gestures[0][0]
        latest_gesture = f"({top_gesture.category_name} {top_gesture.score:.2f})"
        latest_hand_landmarks = result.hand_landmarks
    else:
        latest_gesture = "No Gesture"
        latest_hand_landmarks = None

def run():
    global latest_gesture, latest_hand_landmarks
    base_options = python.BaseOptions(model_asset_path="models/gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=vision.RunningMode.LIVE_STREAM, result_callback=save_result)
    cap = cv.VideoCapture(0)

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv.flip(frame, 1)
            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            frame_timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, frame_timestamp_ms)
            cv.putText(frame, latest_gesture, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            if latest_hand_landmarks:
                for hand_landmarks in latest_hand_landmarks:
                    for landmark in hand_landmarks:
                        h, w, _ = frame.shape
                        cx = int(landmark.x * w)
                        cy = int(landmark.y * h)
                        cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv.imshow('MediaPipe Gesture Recognition', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    run()
