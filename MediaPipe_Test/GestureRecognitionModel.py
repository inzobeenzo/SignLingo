import mediapipe as mp
import cv2
from GestureRecognition import display_batch_of_images_with_gestures_and_hand_landmarks
from GestureRecognitionImages import FILENAMES, HEIGHT, WIDTH, resize_and_show, run_resize_and_show
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if __name__ == "__main__":
    python = mp.tasks
    vision = mp.tasks.vision
    base_options = python.BaseOptions(model_asset_path='MediaPipe_Test/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # run_resize_and_show()

    images, results = [], []
    for img in FILENAMES:
        image = mp.Image.create_from_file(img)
        recognition_result = recognizer.recognize(image)
        if recognition_result.gestures and len(recognition_result.gestures) > 0:
            images.append(image)
            top_gesture = recognition_result.gestures[0][0]
            results.append((top_gesture, recognition_result.hand_landmarks))
        else:
            print(f"No gestures detected in {img}")
    display_batch_of_images_with_gestures_and_hand_landmarks(images, results)