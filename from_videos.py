"""Optional step 1b: turn video files into training examples.

Use this instead of, or as well as, collect.py. Put the videos in folders named after the sign:

    videos/hello/anything.mp4
    videos/thanks/clip1.mov
    videos/thanks/clip2.mp4

then run:

    python from_videos.py --signer dataset

Each video should show ONE sign. A short video (the sign done once) becomes one example.
A long video (the same sign held or repeated) is cut into several 1.5 second examples.
The examples are saved in data/ in the same format collect.py uses, so train.py works as before.
"""
import argparse
import os
import re

import cv2
import numpy as np

from common import CLIP_SECONDS, DATA_DIR, HandTracker

VIDEO_TYPES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def track_video(path):
    """Runs hand tracking on every frame. Returns (hands for each frame, frames per second)."""
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if not 5 <= fps <= 240:                     # some files do not say; assume a normal speed
        fps = 30
    tracker = HandTracker()                     # a fresh tracker for each video
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)              # same mirror as collect.py and live.py
        frames.append(tracker.track(frame, time_ms=len(frames) * 1000 / fps))
    video.release()
    return np.array(frames), fps


def cut_into_clips(hands, fps):
    """Pick the part (or parts) of a tracked video that show the sign."""
    visible = ~np.isnan(hands[:, :, 0, 0]).all(axis=1)          # is any hand in view?
    if not visible.any():
        return []
    first, last = np.flatnonzero(visible)[[0, -1]]
    hands = hands[first:last + 1]                               # drop the empty start and end

    # Many sign videos start and end with the hands resting low. If this one does,
    # keep only the part where the hands are raised.
    wrist_y = hands[:, :, 0, 1]                                 # 0 = top of the picture
    top = np.where(np.isnan(wrist_y), 9.0, wrist_y).min(axis=1)  # highest wrist; 9 = no hand
    lowest, highest = top[top < 9].max(), top.min()
    low = top > lowest - 0.3 * (lowest - highest)               # in the bottom part of its range
    if lowest - highest > 0.15 and low[0] and low[-1]:
        raised = np.flatnonzero(~low)
        hands = hands[raised[0]:raised[-1] + 1]

    length = int(round(CLIP_SECONDS * fps))                     # frames in 1.5 seconds
    if len(hands) <= 2 * length:
        clips = [hands]                                         # short video: one example
    else:                                                       # long video: cut it up
        clips = [hands[i:i + length] for i in range(0, len(hands) - length + 1, length)]

    def has_hands(clip):
        return (~np.isnan(clip[:, :, 0, 0]).all(axis=1)).mean() > 0.3
    return [c for c in clips if len(c) >= 8 and has_hands(c)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="videos", help="folder with one sub-folder per sign")
    parser.add_argument("--signer", default="video",
                        help="a name for who is in these videos, e.g. dataset, youtube, zen")
    args = parser.parse_args()
    signer = re.sub(r"[^a-z0-9]+", "-", args.signer.lower())
    if not os.path.isdir(args.videos):
        raise SystemExit(f"No folder called '{args.videos}'. Make videos/<sign>/ and put videos in.")

    total = 0
    for sign in sorted(os.listdir(args.videos)):
        folder = os.path.join(args.videos, sign)
        if not os.path.isdir(folder):
            continue
        out = os.path.join(DATA_DIR, sign.lower())
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(VIDEO_TYPES):
                continue
            hands, fps = track_video(os.path.join(folder, name))
            clips = cut_into_clips(hands, fps) if len(hands) else []
            stem = re.sub(r"[^A-Za-z0-9]+", "-", os.path.splitext(name)[0])
            os.makedirs(out, exist_ok=True)
            for k, clip in enumerate(clips):
                np.save(os.path.join(out, f"{signer}__{stem}-{k:02d}.npy"), clip)
            total += len(clips)
            note = "" if clips else "   <- no hands found, skipped"
            print(f"{sign}/{name}: {len(hands)} frames -> {len(clips)} example(s){note}")
    print(f"\nSaved {total} examples in {DATA_DIR}/. Now run: python train.py")
