"""Step 2: train the sign classifier and test it fairly.

    python train.py                       test on the LAST examples each person recorded of each sign
    python train.py --test-signer aina    test on a person the model has never seen
    python train.py --mirror              also learn from mirrored copies (for left-handed signers)

Prints accuracy for a simple baseline and for our models, saves confusion_matrix.png,
lists the test clips it got wrong, and saves the model for live.py.
"""
import argparse
from collections import Counter, defaultdict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from common import MODEL_PATH, clip_features, load_clips

parser = argparse.ArgumentParser()
parser.add_argument("--test-signer", help="keep all of this person's clips for testing")
parser.add_argument("--test-share", type=float, default=0.25)
parser.add_argument("--mirror", action="store_true",
                    help="also train on mirrored copies, so signing with either hand works")
args = parser.parse_args()

clips = load_clips()
if not clips:
    raise SystemExit("No data yet. Record some signs with collect.py first.")
counts = Counter(c["sign"] for c in clips)
print("Examples per sign:", dict(counts))
if len(counts) < 2:
    raise SystemExit("Record at least two different signs.")

# ---- split: never test on a clip the model trained on ----
train, test = [], []
if args.test_signer:
    for c in clips:
        (test if c["signer"] == args.test_signer.lower() else train).append(c)
else:
    groups = defaultdict(list)
    for c in clips:
        groups[(c["sign"], c["signer"])].append(c)
    for group in groups.values():
        group.sort(key=lambda c: c["time"])               # oldest first
        n_test = max(1, round(len(group) * args.test_share)) if len(group) > 1 else 0
        train += group[:len(group) - n_test]
        test += group[len(group) - n_test:]
if not train or not test:
    raise SystemExit("Not enough clips to make both a training set and a test set.")
print(f"Training clips: {len(train)}   Test clips: {len(test)}")

# ---- features. Each training clip is also used slightly early and slightly late, because
#      in live use a sign never lines up perfectly with the 1.5 second window. ----
def mirror_clip(clip):
    """The same sign done with the other hand: swap the two hand slots and flip left-right."""
    mirrored = clip[:, ::-1].copy()
    mirrored[..., 0] *= -1
    return mirrored


train_clips = [(c["clip"], c["sign"]) for c in train]
if args.mirror:                 # test clips are never mirrored
    train_clips += [(mirror_clip(clip), sign) for clip, sign in train_clips]
    print(f"Added {len(train)} mirrored copies to the training clips")

SHIFTS = [-0.15, 0.0, 0.15]
X_train = np.array([clip_features(clip, s) for clip, _ in train_clips for s in SHIFTS])
y_train = np.array([sign for _, sign in train_clips for _ in SHIFTS])
X_test = np.array([clip_features(c["clip"]) for c in test])
y_test = np.array([c["sign"] for c in test])

# ---- baseline and models ----
models = {
    "Baseline (always the most common sign)": DummyClassifier(strategy="most_frequent"),
    "Nearest neighbours (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=42),
}
print("\nAccuracy on test clips")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"  {accuracy_score(y_test, model.predict(X_test)):6.1%}  {name}")

forest = models["Random forest"]
predicted = forest.predict(X_test)

# ---- where does it fail? ----
labels = sorted(counts)
ConfusionMatrixDisplay.from_predictions(y_test, predicted, labels=labels, xticks_rotation=45)
plt.title("Random forest on test clips")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

wrong = [(c, p) for c, p in zip(test, predicted) if c["sign"] != p]
print(f"\nMistakes: {len(wrong)} of {len(test)}")
for c, p in wrong[:15]:
    print(f"  signed '{c['sign']}' by {c['signer']}, model said '{p}'   ({c['path']})")

joblib.dump(forest, MODEL_PATH)
print(f"\nSaved {MODEL_PATH}. Now run: python live.py")
