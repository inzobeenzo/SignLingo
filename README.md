# Real-Time Sign & Gesture Recognition

An end-to-end pipeline that recognizes hand signs from a live webcam feed and displays them as text in real time. It uses **MediaPipe** to extract hand landmarks from each frame and a **custom-trained Keras LSTM** to classify the gesture (made by either hand) from a short sequence of those landmarks.

Built as a hands-on project to learn real-time computer vision and temporal sequence modeling.

---

## How it works

Rather than feeding raw pixels to a network, the pipeline works on hand *landmarks* from MediaPipe outputs (the 21 joint x, y, and z coordinates per hand). This turns each frame into a small 126-number vector (21 joints × 3 coordinates × 2 hands) instead of a full image, which keeps the model small and fast.

Because many signs involve motion, the model classifies a **sequence** of frames rather than a single pose. Each sample is a 30-frame window, and an LSTM learns the pattern over time.

The flow is: **collect landmarks → normalize → train LSTM → run live inference.**

---

## Model architecture

The model treats recognition as a sequence-to-category problem, taking input of shape `(30 frames, 126 features)`:

- **LSTM** (64 units, `return_sequences=True`) → LayerNormalization → Dropout (0.5)
- **LSTM** (128 units, `return_sequences=False`) → LayerNormalization → Dropout (0.5)
- **Dense** (64 units, ReLU, L2 regularization)
- **Dense** (softmax, one unit per class)

Trained with the Adam optimizer and categorical cross-entropy. `EarlyStopping` on validation loss prevents overtraining.

**Vocabulary:** 31 classes — 5 word-signs (`hello`, `goodbye`, `thank_you`, `please`, `help`) and the 26 letters of the alphabet.

---

## Repository structure

```text
sign-language-lstm/
├── README.md
├── data.py          # webcam capture → saves landmark sequences as .npy files
├── processing.py    # loads landmarks, applies wrist-relative normalization, builds training arrays
├── model.py         # defines and trains the LSTM, saves the best model
├── detection.py     # live webcam inference with a sliding prediction window
├── best_model.keras # trained model weights
├── models/
│   └── holistic_landmarker.task   # MediaPipe landmark model asset
└── data/            # generated landmark sequences (gitignored)
```

---

## Setup

Requires Python 3.10+. From a virtual environment:

```bash
pip install mediapipe opencv-python tensorflow numpy scikit-learn
```

You also need the MediaPipe landmark model asset (`holistic_landmarker.task`) placed in `models/`.

---

## Usage

The pipeline runs in three stages:

**1. Collect data** — records landmark sequences for each sign in your vocabulary. Repeat for every class (manually sorry...).
```bash
python data.py
```

**2. Train the model** — processes the collected sequences and trains the LSTM, saving the best checkpoint.
```bash
python model.py
```

**3. Run live recognition** — loads the trained model and predicts from your webcam in real time.
```bash
python detection.py
```
Press `q` to release the camera and exit.

---

## Implementation notes

- **Wrist-relative normalization.** Every joint is expressed relative to the wrist, so the same sign produces a similar vector regardless of *where* the hand sits in the frame. This removes translation; it does not remove scale (hand size changes with distance to the camera. See limitations below).
- **Single-hand handling.** When only one hand is visible, the missing hand's 63 features are zero-filled so the input shape stays fixed.
- **Strided inference.** Live prediction runs on a rolling 30-frame window and only every 5th frame, which keeps the loop responsive without re-predicting on every frame.
- **Consistent preprocessing.** The same landmark normalization is applied at both training and inference time, so the model sees data the same way in both.

---

## Scope and limitations

This is a working baseline, not a general sign-language system.

- **Single-signer data.** The training sequences were collected by one person in one environment. The model recognizes that signer's gestures in that setup well; performance on new people, lighting, or backgrounds is minimally tested and, although seemingly consistent, expected to drop. It has not been validated on a held-out signer.
- **No scale normalization yet.** Distance to the camera affects the landmark magnitudes; a reference-length scaling step would make it more robust.
- **Small vocabulary.** 31 classes, and the word-signs are recognized as whole-sequence patterns, not composed grammatically.
- **Not generalized.** Alphabet and language only based on ASL (American Sign Language) and training data may not contain most accurate representations/gestures.

---

## Possible next steps

- Collect data from multiple signers and evaluate on a held-out person to measure real generalization.
- Add scale normalization (divide by a reference hand length).
- Expand the vocabulary and add a confidence-smoothing step to steady live predictions.
- Wrap the inference loop in a web app (browser-side MediaPipe + a small API) for a shareable demo.
