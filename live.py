"""Step 3: live sign to speech.

    python live.py

Sign at the camera. When the model is confident, the word is spoken aloud and added to the
sentence at the bottom.  S = speak the whole sentence, C = clear it, Q = quit.
"""
import argparse
import time
from collections import deque

import cv2
import joblib
import numpy as np

from common import (CLIP_SECONDS, MODEL_PATH, HandTracker, Smoother, clip_features,
                    draw_hands, put_text, speak)

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--confidence", type=float, default=0.6, help="how sure the model must be")
args = parser.parse_args()

model = joblib.load(MODEL_PATH)
tracker = HandTracker()
smoother = Smoother(need=4, min_confidence=args.confidence)
camera = cv2.VideoCapture(args.camera)
if not camera.isOpened():
    raise SystemExit("Could not open the camera. Try --camera 1, and close other apps using it.")

recent = deque()            # (time, hands) for the last CLIP_SECONDS
sentence = []
label, confidence, frame_number = "idle", 0.0, 0

while True:
    ok, frame = camera.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    hands = tracker.track(frame)
    draw_hands(frame, hands)

    now = time.time()
    recent.append((now, hands))
    while now - recent[0][0] > CLIP_SECONDS:
        recent.popleft()

    frame_number += 1
    if frame_number % 3 == 0:                       # predict on every third frame
        clip = np.stack([h for _, h in recent])
        hand_visible = ~np.isnan(clip[:, :, 0, 0]).all(axis=1)
        window_full = now - recent[0][0] > 0.8 * CLIP_SECONDS
        if window_full and hand_visible.mean() > 0.3:
            chances = model.predict_proba([clip_features(clip)])[0]
            label, confidence = model.classes_[chances.argmax()], chances.max()
        else:
            label, confidence = "idle", 1.0
        word = smoother.update(label, confidence)
        if word:
            sentence.append(word)
            speak(word)

    put_text(frame, f"{label}  {confidence:.0%}", 30,
             colour=(0, 255, 0) if confidence >= args.confidence else (180, 180, 180))
    put_text(frame, " ".join(sentence[-8:]), frame.shape[0] - 20, scale=1.0)
    cv2.imshow("Sign to speech   (S = say sentence, C = clear, Q = quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        sentence = []
    if key == ord("s"):
        speak(" ".join(sentence))

camera.release()
cv2.destroyAllWindows()
