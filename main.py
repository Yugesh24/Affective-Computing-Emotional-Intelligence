import pandas as pd
import numpy as np
import cv2
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
# Load FER2013 dataset
df = pd.read_csv('fer2013.csv')
# For UI Control
root =Tk()
root.title('Main Panel')
root.geometry("400x300")
# root.minsize(300, 200)
# root.maxsize(800, 600)
# Extract pixels and labels
pixels = df['pixels'].values
labels = pd.get_dummies(df['emotion']).values

# Convert pixels to images
images = np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape((48, 48)) for pixel in pixels])

# Normalize pixel values
images = images / 255.0

# Split dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape train and test images for CNN input
train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)

# Train the model
model.fit(train_images, train_labels, epochs=1, batch_size=64, validation_data=(test_images, test_labels))

# Load pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open system camera
cap = cv2.VideoCapture(0)



# Define emotion labels
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize variables for emotion ratio
emotion_counts = [0] * len(emotion_labels)

# Create matplotlib figure and axes for emotion ratio chart

fig, ax = plt.subplots()
plt.title('Ratio Window')
ax.set_xlabel('Face Reaction Display Panel')
bar_chart = ax.bar(emotion_labels, emotion_counts)

# Function to update emotion ratio chart
def update_chart():
    for i, bar in enumerate(bar_chart):
        bar.set_height(emotion_counts[i])
    plt.draw()

# Function to start emotion detection
def start_detection():
    global emotion_counts
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract face region and preprocess
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            # Predict emotion
            prediction = model.predict(face_roi)[0]
            max_index = np.argmax(prediction)
            emotion_label = emotion_labels[max_index]


            # Check if the list is empty
            if not emotion_labels:
                print("Emotion labels list is empty.")
            else:
                # Check if max_index is within the range of the list
                if 0 <= max_index < len(emotion_labels):
                    emotion_label = emotion_labels[max_index]
                    # Further code using emotion_label
                else:
                    print("Invalid max_index value:", max_index)

            # Update emotion counts
            emotion_counts[max_index] += 1

            # Draw rectangle around the face and label the emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)

        # Update the emotion ratio chart
        update_chart()

        # Display the frame
        cv2.imshow('Team 7', frame)

        # Exit if 'e' is pressed
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Function to stop emotion detection
def stop_detection():
    root.quit()

# Start and stop buttons
start_button = tk.Button(root, text="Start", command=start_detection,width=10, height=2)
start_button.pack(expand=True, padx=10, pady=10, anchor=tk.CENTER)
# button = tk.Button(frame, text="Click Me", )
# button.pack(expand=True, padx=10, pady=10, anchor=tk.CENTER)
stop_button = tk.Button(root, text="Stop", command=stop_detection,width=10, height=2)
stop_button.pack(expand=True, padx=10, pady=10, anchor=tk.CENTER)
# Display the emotion ratio chart
plt.show()
