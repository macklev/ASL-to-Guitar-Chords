# ASL-to-Guitar-Chords

This project recognizes a small set of American Sign Language hand gestures and maps them to guitar chords. It uses MediaPipe to detect hand landmarks from a webcam feed, trains a k-nearest neighbors model on collected landmark data, then runs live prediction with optional chord sound playback.

## Features

- Collect hand landmark data from a webcam and label it as `A` through `G`
- Train a gesture classifier with scikit-learn
- Run live ASL gesture prediction with smoothing to reduce jitter
- Display the detected letter and mapped chord on screen
- Play a matching chord audio file when a gesture is recognized

## Project Files

- `ASLChordRecognizer.py` - main application for collecting data or running prediction
- `train_model.py` - trains the model from `asl_data.csv` and saves `asl_model.pkl`
- `handtest.py` - simple MediaPipe hand-detection camera test
- `test.py` - basic webcam test
- `sound_test.py` - quick audio playback test
- `asl_data.csv` - collected training data
- `asl_model.pkl` - saved trained model
- `Sounds/` - chord audio files (`A.wav` through `G.wav`)

## Collect Training Data

Run the main app in collection mode to append landmark samples to `asl_data.csv`.

```bash
python ASLChordRecognizer.py
```

When prompted, enter `collect` and then choose a label from `A` to `G`.

Tips:

- Hold one hand clearly in frame
- Collect several samples for each class from slightly different positions and angles
- Keep using the same label for a full capture session

## Train the Model

After collecting enough samples, train the classifier:

```bash
python train_model.py
```

This script:

- loads `asl_data.csv`
- splits the data into training and test sets
- trains a `KNeighborsClassifier`
- prints accuracy and a confusion matrix
- saves the model to `asl_model.pkl`

## Live Prediction

Once the model exists, run the main app again and choose prediction mode:

```bash
python ASLChordRecognizer.py
```

When prompted, enter `predict`.

In prediction mode, the app will:

- detect the hand landmarks
- predict the ASL letter
- smooth predictions over a short history window
- map the letter to a chord name
- play the corresponding audio file

Press `Esc` to exit.

## Audio Files

The code expects chord audio files named `A.wav` through `G.wav`.

Important: the code currently references a folder named `sounds/`, while the repository folder is named `Sounds/`. If you are on a case-sensitive system, either rename the folder to `sounds/` or update the paths in `ASLChordRecognizer.py` and `sound_test.py`.

## Troubleshooting

- If the webcam does not open, try changing `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` in the scripts.
- If no model loads, run `train_model.py` first to create `asl_model.pkl`.
- If audio does not play, verify that `pygame` is installed and the `.wav` files exist in the expected folder.
- If prediction quality is poor, collect more balanced training samples for each class.


