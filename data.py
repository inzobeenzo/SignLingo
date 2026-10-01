import cv2
import mediapipe as mp
import numpy as np
import os
import time
import shutil

cap = cv2.VideoCapture(0)
results = None
actions = ["hello", "goodbye", "thank_you", "please", "help"]
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

delete = False
current_action = "z"
current_sequence = 0
current_frame = 0

# Joint 0: Wrist (This is always the very first x, y, z triplet in your list)
# Joints 1–4: The Thumb (starting from the base base moving to the tip)
# Joints 5–8: The Index Finger (knuckle to the tip)
# Joints 9–12: The Middle Finger
# Joints 13–16: The Ring Finger 
# Joints 17–20: The Pinky Finger (ending at the very tip of your pinky) 
def print_result_callback(result, output_image, timestamp_ms):
    global results, current_action, current_sequence, current_frame
    left_features = []
    right_features = []
    if result.left_hand_landmarks:
        for landmark in result.left_hand_landmarks:
            left_features.append(landmark.x)
            left_features.append(landmark.y)
            left_features.append(landmark.z)
    else:
        left_features = [0.0] * 63
    if result.right_hand_landmarks:
        for landmark in result.right_hand_landmarks:
            right_features.append(landmark.x)
            right_features.append(landmark.y)
            right_features.append(landmark.z)
    else:
        right_features = [0.0] * 63

    final_features = np.array(left_features + right_features).flatten()

    if delete and os.path.exists("data"):
        shutil.rmtree(f"data/{str(current_action)}", ignore_errors=True)
    
    dirs = os.path.join("data", current_action, str(current_sequence))
    os.makedirs(dirs, exist_ok=True)
    file_path = os.path.join(dirs, f"{current_frame}.npy")
    np.save(file_path, final_features)
    current_frame += 1

options = mp.tasks.vision.HolisticLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="models/holistic_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    min_hand_landmarks_confidence=0.5,
    # min_face_detection_confidence=0.5,
    # min_pose_landmarks_confidence=0.5,
    result_callback=print_result_callback
)

frame_timestamp = 0
num_frames = 30
sequence_length = 30

if __name__ == "__main__":
    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as holistic_landmarker:
        while True:
            ret, frame = cap.read()
            vid = cv2.flip(frame, 1)
            if not ret:
                print("Couldn't grab frame")
                break

            cv2.imshow('Default Webcam', vid)
            vidCol = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=vidCol)

            frame_timestamp += 1
            holistic_landmarker.detect_async(mp_image, frame_timestamp)

            if current_frame == 30:
                cap.release()
                current_frame = 0
                current_sequence += 1
                print(current_sequence)
                time.sleep(0.5)
                cap = cv2.VideoCapture(0)

            if current_sequence == 30:
                cap.release()
                cv2.destroyAllWindows
                break

            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break