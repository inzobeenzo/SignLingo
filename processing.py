import os
import numpy as np
from data import actions, current_action, current_sequence, current_frame

sequences = []
labels = []
data = {"hello": 0, "goodbye": 1, "thank you": 2, "please": 3, "help": 4}
X_data = []

for action in actions:
    for sequence in range(30):
        window = []
        for frame_num in range(30):
            try:
                file_path = os.path.join("data", action, str(sequence), f"{frame_num}.npy")
                res = np.load(file_path)
                window.append(res)
            except FileNotFoundError:
                print("fart")
                pass
        X_data.append(window)

print(X_data)

