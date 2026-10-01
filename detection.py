import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
from processing import data
from data import options

model = load_model("best_model.keras")
data = {v: k for k, v in data.items()}

sequence = []
threshold = 0.5
latest_prediction = "None"
latest_confidence = 0.0

def extract_keypoints(result):
    if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
        hand_data = result.left_hand_landmarks[0]
        lh_matrix = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks])
        left_wrist = lh_matrix[0]
        lh = (lh_matrix - left_wrist).flatten()
    else:
        lh = np.zeros(21 * 3)

    if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
        hand_data = result.right_hand_landmarks[0]
        rh_matrix = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks])
        right_wrist = rh_matrix[0]
        rh = (rh_matrix - right_wrist).flatten()
    else:
        rh = np.zeros(21 * 3)
        
    return np.concatenate([lh, rh])

def print_result_callback(result, output_image, timestamp_ms):
    global sequence, latest_prediction, latest_confidence

    left_missing = not result.left_hand_landmarks or len(result.left_hand_landmarks) == 0
    right_missing = not result.right_hand_landmarks or len(result.right_hand_landmarks) == 0
    
    if left_missing and right_missing:
        sequence = []
        latest_prediction = "None"
        latest_confidence = 0.0
        return
    
    keypoints = extract_keypoints(result)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30 and frame_count % 5 == 0:
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        res = model(input_data, training=False).numpy()[0]
        idx = int(np.argmax(res)) 

        if res[idx] > threshold:
            latest_prediction = data[idx] 
            latest_confidence = res[idx]
        else:
            latest_prediction = "None"
            latest_confidence = 0.0

options.result_callback = print_result_callback
cap = cv2.VideoCapture(0)
frame_count = 0
with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Couldn't grab frame")
            break

        flipped = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        frame_count += 1
        timestamp = int(time.time() * 1000) + frame_count
        landmarker.detect_async(mp_image, timestamp)

        if latest_prediction != "None":
            cv2.rectangle(flipped, (0, 0), (640, 45), (245, 117, 16), -1)
            cv2.putText(flipped, f'Gesture: {latest_prediction} ({latest_confidence:.2f})', 
            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
             cv2.putText(flipped, 'Scanning...', (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Live Gesture Recognition', flipped)

        if cv2.waitKey(1) == ord('q'):
            break