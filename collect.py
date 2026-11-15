"""Step 1: record examples of one sign.

    python collect.py --sign hello --signer zen
    python collect.py --sign idle  --signer zen     (hands resting, moving about, or out of view)

Press SPACE to record one example, U to undo the last one, Q to quit.
Each example is saved automatically. Only hand point coordinates are saved, not video.
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import CLIP_SECONDS, DATA_DIR, HandTracker, draw_hands, put_text

GET_READY = 0.7     # seconds between pressing SPACE and the recording starting

parser = argparse.ArgumentParser()
parser.add_argument("--sign", required=True, help="the word this sign means, e.g. hello")
parser.add_argument("--signer", required=True, help="first name of the person signing")
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()
if "__" in args.signer:
    parser.error("signer name cannot contain '__'")

folder = os.path.join(DATA_DIR, args.sign.lower())
os.makedirs(folder, exist_ok=True)
saved = len([f for f in os.listdir(folder) if f.endswith(".npy")])

tracker = HandTracker()
camera = cv2.VideoCapture(args.camera)
if not camera.isOpened():
    raise SystemExit("Could not open the camera. Try --camera 1, and close other apps using it.")

started = None      # time SPACE was pressed, or None when waiting
frames = []
this_session = []   # files saved since this window opened, so U can undo them
while True:
    ok, frame = camera.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)                  # mirror, so it feels like looking in a mirror
    hands = tracker.track(frame)
    draw_hands(frame, hands)
    now = time.time()

    if started is None:
        put_text(frame, f"'{args.sign}'  saved: {saved}", 30)
        put_text(frame, "SPACE = record   U = undo last   Q = quit", 60, scale=0.6)
    elif now < started + GET_READY:
        put_text(frame, "Get ready...", 30, colour=(0, 255, 255))
    elif now < started + GET_READY + CLIP_SECONDS:
        frames.append(hands)
        put_text(frame, "RECORDING - sign now", 30, colour=(0, 0, 255))
    else:
        if len(frames) >= 8:
            name = f"{args.signer.lower()}__{time.strftime('%Y%m%d-%H%M%S')}-{saved:03d}.npy"
            np.save(os.path.join(folder, name), np.stack(frames))
            this_session.append(os.path.join(folder, name))
            saved += 1
        else:
            print("Camera was too slow for that one, try again.")
        started, frames = None, []

    cv2.imshow("Collect signs", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" ") and started is None:
        started, frames = now, []
    if key == ord("u") and started is None and this_session:
        os.remove(this_session.pop())               # delete the most recent recording
        saved -= 1

camera.release()
cv2.destroyAllWindows()
print(f"{saved} examples of '{args.sign}' in {folder}")
