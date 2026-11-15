"""Shared pieces used by collect.py, train.py and live.py.

1. HandTracker   : finds 21 points on each hand in a camera frame (MediaPipe, a pretrained model)
2. clip_features : turns a short clip of hand points into one row of numbers for our classifier
3. Smoother      : decides when the live prediction is steady enough to say out loud
4. speak         : text to speech using the voice built into the computer
"""
import os
import platform
import subprocess
import time
import urllib.request

import numpy as np

CLIP_SECONDS = 1.5      # every sign is recorded as a clip this long
N_FRAMES = 12           # each clip is squeezed to this many frames before it goes into the model
DATA_DIR = "data"       # data/<sign>/<signer>__<time>.npy
MODEL_PATH = "sign_model.joblib"

HAND_MODEL_FILE = "hand_landmarker.task"
HAND_MODEL_URLS = [     # the second address is a spare in case the first one moves
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
]


def download_hand_model():
    def complete(path):                      # a broken download is tiny; the real file is ~8 MB
        return os.path.exists(path) and os.path.getsize(path) > 1_000_000
    if complete(HAND_MODEL_FILE):
        return
    print("Downloading the hand tracking model (once, about 8 MB)...")
    for url in HAND_MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, HAND_MODEL_FILE + ".part")
            if complete(HAND_MODEL_FILE + ".part"):
                os.replace(HAND_MODEL_FILE + ".part", HAND_MODEL_FILE)
                return
        except OSError:
            pass
    raise SystemExit("Could not download the hand tracking model. Check your internet, or download "
                     f"it in a browser from\n{HAND_MODEL_URLS[0]}\nand put it in this folder.")

# which of the 21 hand points to join with lines when drawing
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]


# ---------------------------------------------------------------- hand tracking
class HandTracker:
    """Wraps MediaPipe's pretrained Hand Landmarker. We did not train this part."""

    def __init__(self):
        import mediapipe as mp

        download_hand_model()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_FILE),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.mp = mp
        self.last_ms = 0

    def track(self, frame_bgr, time_ms=None):
        """Returns an array of shape (2 hands, 21 points, 3 coordinates).

        A hand that is not visible is filled with NaN. x and y are measured in units of the
        frame HEIGHT, so a hand keeps its shape whatever the camera's width is.
        time_ms is the frame's time in a video file; leave it out for a live camera.
        """
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if time_ms is None:
            time_ms = time.monotonic() * 1000
        self.last_ms = max(self.last_ms + 1, int(time_ms))      # must always increase
        result = self.detector.detect_for_video(image, self.last_ms)

        hands = np.full((2, 21, 3), np.nan, dtype=np.float32)
        for points, handedness in zip(result.hand_landmarks, result.handedness):
            side = getattr(handedness[0], "category_name", "")       # "Left" or "Right"
            slot = 0 if side == "Left" else 1
            if not np.isnan(hands[slot, 0, 0]):       # slot already taken: use the other one
                slot = 1 - slot
            hands[slot] = [[p.x * w / h, p.y, p.z * w / h] for p in points]
        return hands


def draw_hands(frame, hands):
    import cv2
    h = frame.shape[0]
    for hand, colour in zip(hands, [(0, 200, 255), (255, 160, 0)]):
        if np.isnan(hand[0, 0]):
            continue
        pts = [(int(x * h), int(y * h)) for x, y, _ in hand]
        for a, b in HAND_LINES:
            cv2.line(frame, pts[a], pts[b], colour, 2)
        for p in pts:
            cv2.circle(frame, p, 4, (255, 255, 255), -1)


def put_text(frame, text, y, scale=0.8, colour=(255, 255, 255)):
    import cv2
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


# ---------------------------------------------------------------- features
def fix_hand_swaps(clip):
    """MediaPipe sometimes changes its mind about whether a hand is the left or the right one.

    In frames where only ONE hand is visible, put it in the slot it uses most often in this
    clip, so a one-handed sign does not jump between the two slots.
    """
    seen = ~np.isnan(clip[:, :, 0, 0])                 # (frames, 2 hands)
    alone = seen.sum(axis=1) == 1
    if alone.any():
        main = int(seen[alone, 1].sum() >= seen[alone, 0].sum())
        wrong = alone & ~seen[:, main]
        clip[wrong] = clip[wrong][:, ::-1]
    return clip


def clip_features(clip, shift=0.0):
    """Turn one clip, shape (frames, 2, 21, 3), into a single row of numbers.

    For each hand and each of N_FRAMES evenly spaced frames we keep:
      * hand SHAPE: the 21 points measured from the wrist and divided by the hand's size,
        so it does not matter where the hand is or how close it is to the camera
      * hand MOVEMENT: where the wrist is compared with its average position in the clip
      * whether the hand was visible at all
    `shift` slides the sampling window earlier/later (used to make extra training examples).
    """
    clip = fix_hand_swaps(np.array(clip, dtype=np.float32))
    n = len(clip)
    positions = np.linspace(0, n - 1, N_FRAMES) + shift * n
    frames = clip[np.clip(np.round(positions).astype(int), 0, n - 1)]

    parts = []
    for slot in range(2):
        hand = frames[:, slot]                               # (N_FRAMES, 21, 3)
        seen = ~np.isnan(hand[:, 0, 0])
        shape = np.zeros((N_FRAMES, 63), dtype=np.float32)
        move = np.zeros((N_FRAMES, 2), dtype=np.float32)
        if seen.any():
            from_wrist = hand[seen] - hand[seen][:, :1]
            size = np.linalg.norm(from_wrist[:, 9, :2], axis=1) + 1e-6   # wrist to middle knuckle
            shape[seen] = (from_wrist / size[:, None, None]).reshape(-1, 63)
            wrist = hand[seen][:, 0, :2]
            move[seen] = (wrist - wrist.mean(axis=0)) / np.median(size)
        parts += [shape.ravel(), move.ravel(), seen.astype(np.float32)]
    return np.concatenate(parts)


def load_clips():
    """Reads every saved clip. Returns a list of dicts: sign, signer, time, path, clip."""
    clips = []
    if not os.path.isdir(DATA_DIR):
        return clips
    for sign in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".npy"):
                signer, stamp = name[:-4].split("__")
                path = os.path.join(folder, name)
                clips.append({"sign": sign, "signer": signer, "time": stamp,
                              "path": path, "clip": np.load(path)})
    return clips


# ---------------------------------------------------------------- live decision
class Smoother:
    """Only announce a sign when the model gives the same confident answer several times in a row.

    The same sign is not repeated until the signer pauses (an "idle" stretch) in between.
    """

    def __init__(self, need=4, min_confidence=0.6):
        self.need, self.min_confidence = need, min_confidence
        self.current, self.count, self.last_spoken = "idle", 0, None

    def update(self, label, confidence):
        if confidence < self.min_confidence:
            label = "idle"                      # not sure enough: treat as nothing
        if label == self.current:
            self.count += 1
        else:
            self.current, self.count = label, 1
        if self.count < self.need:
            return None
        if label == "idle":
            self.last_spoken = None             # a pause lets the same sign be said again
            return None
        if label != self.last_spoken:
            self.last_spoken = label
            return label
        return None


# ---------------------------------------------------------------- speech
def speak(text):
    """Say text out loud without freezing the camera window. Uses the computer's own voice."""
    text = "".join(ch for ch in text if ch.isalnum() or ch in " ,.?!-")
    if not text.strip():
        return
    system = platform.system()
    try:
        if system == "Windows":
            command = ["powershell", "-NoProfile", "-Command",
                       "Add-Type -AssemblyName System.Speech; "
                       f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"]
        elif system == "Darwin":
            command = ["say", text]
        else:
            command = ["espeak", text]
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        print("(could not find a text-to-speech voice on this computer) ->", text)
