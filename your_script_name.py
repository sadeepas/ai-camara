import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

# Load pre-trained models
emotion_model = load_model('path_to_emotion_model.h5')
gesture_model = load_model('path_to_gesture_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the main window
root = Tk()
root.title("AI Camera App")
root.geometry("800x600")

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Create a label to display the video
video_label = Label(root)
video_label.pack()

# Function to process each frame
def update_frame():
    ret, frame = cap.read()
    if ret:
        # Convert the frame to grayscale for emotion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            
            # Predict the emotion
            emotion_prediction = emotion_model.predict(roi_gray)[0]
            max_index = int(np.argmax(emotion_prediction))
            emotion_label = emotion_labels[max_index]
            
            # Draw the rectangle around the face and label the emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Convert the frame to ImageTk format
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    video_label.after(10, update_frame)

# Start the video loop
update_frame()

# Start the GUI event loop
root.mainloop()

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
