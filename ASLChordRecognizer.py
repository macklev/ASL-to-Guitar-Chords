import cv2
import mediapipe as mp
import numpy as np
import csv
import joblib
from collections import deque

class ASLChordRecognizer:
    def __init__(self, mode="predict"):
        self.mode = mode

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Smoothing
        self.history = deque(maxlen=5)

        # Model (only load in predict mode)
        self.model = None
        if self.mode == "predict":
            try:
                self.model = joblib.load("asl_model.pkl")
            except:
                print("No model found. Run training first.")

        # Chord mapping
        self.chord_map = {
            "A": "A Major",
            "B": "B Major",
            "C": "C Major",
            "D": "D Major",
            "E": "E Major",
            "F": "F Major",
            "G": "G Major"
        }

    # -----------------------------
    def preprocess_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, rgb

    # -----------------------------
    def extract_landmarks(self, results):
        if not results.multi_hand_landmarks:
            return None, None

        hand_landmarks = results.multi_hand_landmarks[0]

        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])

        coords = np.array(coords)

        # Normalize relative to wrist
        coords = coords - coords[0]

        features = coords.flatten()

        return coords, features

    # -----------------------------
    def collect_data(self, features, label):
        with open("asl_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(features) + [label])

    # -----------------------------
    def predict_letter(self, features):
        if self.model is None or features is None:
            return None
        return self.model.predict([features])[0]

    # -----------------------------
    def smooth_prediction(self, letter):
        if letter is None:
            return None
        self.history.append(letter)
        return max(set(self.history), key=self.history.count)

    # -----------------------------
    def get_chord(self, letter):
        if letter is None:
            return None
        return self.chord_map.get(letter, "Unknown")

    # -----------------------------
    def draw_output(self, frame, results, letter, chord, mode, label=None):
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, handLms, self.mp_hands.HAND_CONNECTIONS
                )

        if mode == "collect":
            cv2.putText(frame, f"Collecting: {label}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if letter:
            cv2.putText(frame, f"Letter: {letter}", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        if chord:
            cv2.putText(frame, chord, (50,150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        return frame

    # -----------------------------
    def run(self):
        cap = cv2.VideoCapture(1)  # adjust if needed

        label = None
        if self.mode == "collect":
            label = input("Enter label (A-G): ")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame, rgb = self.preprocess_frame(frame)
            results = self.hands.process(rgb)

            coords, features = self.extract_landmarks(results)

            letter = None
            chord = None

            if self.mode == "collect":
                if features is not None:
                    self.collect_data(features, label)

            elif self.mode == "predict":
                letter = self.predict_letter(features)
                letter = self.smooth_prediction(letter)
                chord = self.get_chord(letter)

            frame = self.draw_output(frame, results, letter, chord, self.mode, label)

            cv2.imshow("ASL Chord Recognizer", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# -----------------------------
if __name__ == "__main__":
    mode = input("Enter mode (collect/predict): ")
    app = ASLChordRecognizer(mode=mode)
    app.run()