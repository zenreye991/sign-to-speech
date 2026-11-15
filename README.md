# Sign to speech, speech to captions

A two-way bridge between a deaf signer and a hearing person who does not sign. One half watches
someone sign and says the words out loud; the other listens to the reply and puts it on screen
in large captions.

**Track:** AI + Languages · **Team:** Zen Rey, Artem, Kai Jian · International School ParkCity
**Demo video:** https://drive.google.com/file/d/1jdufwIJnZwm9_49IhE03wXSom_mSE1Rh/view?usp=sharing

---

## The problem

A deaf person who signs and a hearing person who does not have no shared language in the
ordinary situations where they meet: a shop counter, a clinic reception, a school office.
Writing notes works, but it is slow, it needs paper or a phone in hand, and it falls apart
when the other person is busy. Interpreters exist, but not at a counter at 9am.

Our user is a deaf student at our own school who needs to ask a hearing member of staff for
something simple and immediate. The moment they would reach for this tool is the moment they
realise the person in front of them does not sign.

## What we built

Two halves of one conversation, each runnable on an ordinary laptop.

**Signs to speech** (the Python scripts here, a model we trained). A webcam watches someone sign
and the word is spoken aloud.

- **In:** live video of a person signing. **Out:** the English word through the speaker, plus a
  running sentence on screen.
- The signer signs, pausing briefly between signs, and each recognised sign is spoken
  immediately. Pressing **S** speaks the whole sentence again for someone who missed it.
- It currently knows six signs in American Sign Language: **hello, yes, no, thanks, please** and
  **water**, plus a seventh class, `idle`, for when the person is not signing.

**Speech to captions** (`speech-to-text/`, built on a pretrained model). The hearing person
replies out loud and their words appear as large on-screen text.

- **In:** live microphone audio. **Out:** captions at 26px, a new line each time the speaker
  pauses.
- Audio is collected in half-second chunks. A pause of 0.8 seconds closes the current line and
  starts the next, so captions read as sentences rather than a wall of text.

## What we added

Our own idea, built from scratch this weekend. The parts we think are worth pointing at:

- **An `idle` class.** Most tutorials classify only signs, so the model must label every moment
  as *some* sign. We recorded 90 clips of not-signing — hands resting, scratching, lifting into
  position, holding a phone — so the system knows when to stay quiet. This turned out to be the
  hardest class and the source of almost all our errors.
- **Our own data, from our own camera.** 390 clips recorded by the team, no public dataset.
- **A fair split by time.** We train on earlier clips and test on the most recent 25% of each
  sign, never shuffling clips between the two.
- **A left-handed option.** `--mirror` adds a flipped copy of every training clip, so a
  left-handed signer is recognised without recording the whole vocabulary again.
- **Privacy by design.** Only hand coordinates are saved, never video or images, so no faces
  are stored anywhere in the dataset.

## How it works

**Data.** 390 clips of 1.5 seconds each, recorded on a laptop webcam by two of us:
hello 60, yes 60, no 60, thanks 60, please 30, water 30, and 90 `idle`. Hello, yes, no and
thanks were recorded by both signers (30 each); please, water and idle by one. We varied distance,
position and speed between recordings. Each clip is stored as the coordinates of 21 points on
each hand for every frame — 288 clips for training (576 after mirroring) and 102 for testing.

**Features.** For each clip we take 12 evenly spaced frames. For each hand in each frame we keep
its *shape* (the 21 points measured from the wrist and divided by hand size, so where the hand
is and how close it is to the camera do not matter), the wrist's *movement* relative to its
average position, and whether the hand was visible at all. That gives 1,584 numbers per clip.

**Model.** A random forest (scikit-learn, 300 trees), compared against k-nearest-neighbours and
a baseline that always predicts the most common class. Hand tracking itself is Google's
pretrained MediaPipe Hand Landmarker; we did not train that part.

**Deciding when to speak.** A single frame is not enough. The live program only speaks after
four consecutive confident predictions of the same sign, and will not repeat a word until the
signer pauses. In live testing this was enough on its own: we did not need to raise the
confidence threshold above its default of 0.6.

**The speech half.** OpenAI's Whisper (`small`) turns the reply into text. We did not train it;
the work on this side is making a batch model behave like a live one. Microphone audio arrives
in half-second chunks and the phrase being spoken is re-transcribed twice a second, so the
caption updates while the person is still talking. We measure the loudness of each 0.1-second
slice: if the last 0.8 seconds are below the silence threshold, the phrase is finished and the
line is locked in; if someone talks for 15 seconds without pausing, everything but the last
sentence is locked in and the rest carries over. Stretches of pure silence are thrown away
rather than sent to Whisper, because it tends to invent words when given silence.

## Results

Tested on 102 clips the model never saw during training:

| Model | Accuracy |
|---|---|
| Baseline (always the most common class) | 21.6% |
| Nearest neighbours (k=3) | 90.2% |
| **Random forest** | **98.0%** |

**Where it fails.** 2 mistakes out of 102:

- One `idle` clip was read as "please", which is one of the two signs we have only 30 examples
  of. Almost every error we saw across our training runs was of this kind: a non-signing moment
  mistaken for a sign. No sign was ever confused with a different sign. Earlier runs mistook several
  idle clips for "hello", which we think is because raising a hand near the head to scratch or
  adjust hair looks much like the hello salute. We fixed most of this by recording many more
  idle clips of exactly those in-between movements, taking accuracy from 87% to 98%.
- One "thanks" clip was read as `idle`, and the same clip was misread in every run. We think the
  hand never came fully into view.

### A harder test: a signer the model had never seen

The test above holds back the newest clips of each sign, but they come from the same recording
sessions as the training clips — same room, same lighting, same person. So we ran a second test:
train on one signer only, then test on all 120 clips from the other, whom the model had never
seen. Only four of our signs (hello, yes, no, thanks) were recorded by both of us, so that test
covers those four.

| Model | Same signer (102 clips) | New signer (120 clips) |
|---|---|---|
| Baseline (most common class) | 21.6% | 0.0% \* |
| Nearest neighbours (k=3) | 90.2% | **98.3%** |
| Random forest | **98.0%** | 85.0% |

\* The most-frequent baseline always predicts `idle`, and the second signer recorded no `idle`
clips, so it scores zero. Chance across the four signs in that test is 25%.

**The two models swap places.** The random forest, clearly the best on the first test, was the
worse of the two on an unseen signer, while nearest neighbours went the other way. The forest had
learned details specific to one person's hands. Testing only on held-out clips from the same
signer would have hidden this completely, and it is the result we were least expecting.

**Where the forest failed.** Fifteen of its 18 errors (the ones the program lists) were a single
confusion: the new signer's "hello" read as `idle`. We think this is a side effect of our own
earlier fix. Many of our `idle` clips show a hand being raised near the head, which is close to
what the hello salute looks like. Adding them removed our false alarms, but it also taught the
model that a hand near the head means "not signing", and the second signer's hello falls into
that gap. Fixing false alarms cost us a real sign for a person the model had not met.

**The speech half is not measured.** We tested it by speaking into it and reading the captions,
which worked, but we did not compute a word error rate against a labelled transcript. We are
reporting that honestly rather than guessing a number. On a CPU the captions lag a few seconds
behind the voice; on a GPU they keep up.

**What it cannot do.** The sign half recognises single signs, not sign language. It has no grammar, no
facial expression (a grammatical part of ASL), no fingerspelling, and it cannot follow fast
continuous signing. It is a prototype, not an interpreter.

## What we'd do next

1. **Record many more signers.** Two is not enough, and our second test shows why: the model
   latches onto one person's hands. We would also have each signer record `idle`, which only one
   of us did, and re-check whether nearest neighbours still wins once more people are in.
2. **Test it with deaf signers.** None of us signs fluently. The people who would use this are
   the ones who should judge whether the signs are right and whether it helps.
3. **Put it on a phone.** MediaPipe runs on phones, and our model is small enough. A phone held
   by the listener is far more useful than a laptop.
4. **Even out the dataset.** "Please" and "water" have 30 examples each against 60 for the
   others, and our one sign-related error involved the smaller end of that range.
5. **Handle position, not just hand shape.** Our features deliberately ignore where the hand is,
   which is why we avoided sign pairs that differ only by body position, like "mother" and
   "father". Adding the face as a reference point would unlock a much larger vocabulary.

## Sources

- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
  (Google) — pretrained hand tracking model
- [Whisper `small`](https://github.com/openai/whisper) (OpenAI) — pretrained speech recognition
  model, used as-is for the captions half
- [scikit-learn](https://scikit-learn.org/) — random forest, k-NN, baseline, metrics
- [Gradio](https://gradio.app/) — the captions interface
- OpenCV, NumPy, SciPy, Matplotlib, ffmpeg
- Sign references: (20+ Basic Sign Language Phrases for Beginners | ASL), (Learn How to Sign Water in ASL | LearnHowToSign.org)
- Code written with the help of Claude (Anthropic)
- Dataset: recorded by the team. No public dataset was used.

---

# Running it yourself

## The captions half

Open `speech-to-text/speech_to_text.ipynb` in Google Colab and choose Runtime → Run all. For
captions that keep up with the speaker, first set Runtime → Change runtime type → T4 GPU. The
last cell prints a link to the live captions page.

## The sign half: set up (once)

1. Install Python from python.org. Any version from 3.9 to 3.14 works.
2. In a terminal in this folder:

   ```
   pip install -r requirements.txt
   ```

3. Check it installed:

   ```
   python -c "import mediapipe, cv2, sklearn, matplotlib; print('ready, mediapipe', mediapipe.__version__)"
   ```

On a Mac, type `python3` wherever this file says `python`. The first run downloads the hand
tracking model (about 8 MB), so you need internet once.

## Record signs

```
python collect.py --sign hello --signer zen
```

A camera window opens with your hand points drawn on it. Press **SPACE**, wait for "RECORDING",
and do the sign once. Repeat 30 or more times; each one saves automatically. **U** deletes the
last one, **Q** quits. Record a class called `idle` too, doing anything that is not a sign.

You can also build examples from video files instead — put them in `videos/<sign>/` and run
`python from_videos.py --signer dataset`.

## Train

```
python train.py --mirror
```

Prints the baseline and model accuracy, lists every test clip it got wrong, and saves
`confusion_matrix.png` and `sign_model.joblib`. To test on a person the model has never seen:

```
python train.py --test-signer aina --mirror
```

## Run it live

```
python live.py
```

Sign at the camera, pausing between signs. **S** speaks the whole sentence, **C** clears it,
**Q** quits. If it speaks too readily, raise the bar with `--confidence 0.75`.

## Files

| File | What it does |
|---|---|
| `collect.py` | records examples of one sign from the webcam |
| `from_videos.py` | makes examples from video files instead |
| `train.py` | trains, tests against a baseline, saves the model |
| `live.py` | camera to speech |
| `common.py` | hand tracking, features, smoothing, speech |
| `data/` | our recorded clips (hand coordinates only) |
| `speech-to-text/speech_to_text.ipynb` | live Whisper captions (Colab) |

## If something goes wrong

- **Camera will not open:** close other apps using it, or try `--camera 1`. On a Mac, allow
  Terminal to use the camera in System Settings → Privacy & Security.
- **"Could not download the hand tracking model":** paste the `curl` line the message shows,
  then run your command again.
- **No voice:** Windows and Mac have one built in. On Linux, install `espeak`.
- **It speaks when you are not signing:** record more `idle` examples of those in-between
  movements, retrain, or raise `--confidence`.
