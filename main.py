import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Load FER2013 dataset
data = pd.read_csv('fer2013.csv')

# Split the data into training and testing sets
# (Assuming you have already done this during model training)

# Load pre-trained emotion detection model
model = tf.keras.models.load_model('D:\Team-7 Project\Project Demo\emotion_detection_model.h5')

# Map emotion labels to corresponding integers
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                  4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Create Tkinter GUI
class FacialEmotionDetectionApp:
    def __init__(self, master):
        self.master = master
        self.video_source = 0  # Set video source to default webcam
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(master, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(master, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT)

        self.detecting_label = tk.Label(master, text="")
        self.detecting_label.pack()

        self.emotion_label = tk.Label(master, text="")
        self.emotion_label.pack()

        self.is_detecting = False
        self.delay = 15
        self.emotion_report = None  # Store emotion report
        self.update()

    def start(self):
        self.is_detecting = True
        self.emotion_report = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0,
                               'Sad': 0, 'Surprise': 0, 'Neutral': 0}
        self.detecting_label.config(text="Detecting")

    def stop(self):
        self.is_detecting = False
        self.detecting_label.config(text="Not Detecting")
        self.generate_report()

    def update(self):
        ret, frame = self.vid.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                print("Error: Unable to load the cascade classifier.")
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            detected_emotion = None

            # Draw rectangles around the faces and display detected emotion
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi = gray[y:y+h, x:x+w]

                # Resize face for model input
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = np.expand_dims(face_roi, axis=-1)

                # Perform emotion detection inference
                predictions = model.predict(face_roi)
                emotion_prediction = np.argmax(predictions)

                detected_emotion = emotion_labels[emotion_prediction]

                if self.is_detecting:  # Update report only if detecting
                    self.emotion_report[detected_emotion] += 1

                # Display detected emotion on the frame
                cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Display the video feed
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.master.after(self.delay, self.update)

    def generate_report(self):
        if self.emotion_report is not None:
            plt.figure(figsize=(8, 6))
            emotions, counts = zip(*self.emotion_report.items())
            plt.bar(emotions, counts, color='skyblue')
            plt.xlabel('Emotion')
            plt.ylabel('Count')
            plt.title('Emotion Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Create the GUI window
root = tk.Tk()
root.title('Affective Computing - Emotion AI')
app = FacialEmotionDetectionApp(root)
root.mainloop()
