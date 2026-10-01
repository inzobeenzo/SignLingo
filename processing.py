import os
import numpy as np

data = {"hello": 0, "goodbye": 1, "thank_you": 2, "please": 3, "help": 4, "a": 5, "b": 6, 
        "c": 7, "d": 8, "e": 9, "f": 10, "g": 11, "h": 12, "i": 13, "j": 14, "k": 15, 
        "l": 16, "m": 17, "n": 18, "o": 19, "p": 20, "q": 21, "r": 22, "s": 23, "t": 24, 
        "u": 25, "v": 26, "w": 27, "x": 28, "y": 29, "z": 30}

def process_data():
    X_data = []
    Y_data = []
    for action in data.keys():
        print(f"Processing: {action}")
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
            Y_data.append(data[action]) 

    X_data = np.array(X_data)

    X_reshaped = X_data.reshape(-1, 30, 2, 21, 3)
    wrists = X_reshaped[:, :, :, 0:1, :]
    X_normalized = X_reshaped - wrists
    X_final = X_normalized.reshape(-1, 30, 126)

    return X_final, np.array(Y_data)


