import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import datetime
import time

# --- Configuration & Imports ---
APP_WIDTH = 1100
APP_HEIGHT = 750
BG_COLOR = "#202124"
ACCENT_COLOR = "#8ab4f8"
TEXT_COLOR = "#e8eaed"

# Try importing AI libraries gracefully
TF_AVAILABLE = False
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not installed. AI features disabled.")

class SmartCameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.window.configure(bg=BG_COLOR)

        # --- State Variables ---
        self.camera_index = 0
        self.is_running = True
        self.current_filter = "Normal"
        self.detect_emotion = False
        self.show_fps = True
        
        # Optimization: Frame Skipping
        self.frame_count = 0
        self.process_every_n_frames = 5 # Run AI every 5th frame
        self.last_emotions = []         # Cache for skipped frames
        self.prev_time = 0              # For FPS calculation

        # --- AI & CV Setup ---
        self.emotion_model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load Haar Cascade (Face Detector)
        # Tries to find local file first, falls back to OpenCV system file
        local_cascade = 'haarcascade_frontalface_default.xml'
        if os.path.exists(local_cascade):
            self.face_classifier = cv2.CascadeClassifier(local_cascade)
        else:
            self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.load_ai_model()
        self.init_ui()
        
        # Start Camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open video source.")
            self.window.destroy()
            return
        
        self.update_frame()

    def load_ai_model(self):
        if not TF_AVAILABLE:
            return
        
        model_path = 'emotion_model.h5'
        if os.path.exists(model_path):
            try:
                self.emotion_model = load_model(model_path)
                print(f"SUCCESS: Loaded {model_path}")
            except Exception as e:
                print(f"ERROR: Could not load model. {e}")
        else:
            print(f"WARNING: {model_path} not found. Emotion features disabled.")

    def init_ui(self):
        # 1. Header
        header = tk.Frame(self.window, bg=BG_COLOR)
        header.pack(fill=tk.X, pady=10)
        tk.Label(header, text="AI SMART LENS", font=("Roboto", 22, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR).pack()

        # 2. Main Layout (Video Left, Controls Right)
        container = tk.Frame(self.window, bg=BG_COLOR)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Video Frame
        self.video_frame = tk.Label(container, bg="black", bd=2, relief="solid")
        self.video_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Controls Sidebar
        sidebar = tk.Frame(container, bg="#303134", width=300)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))

        # --- Filter Section ---
        lbl_filt = tk.Label(sidebar, text="VISUAL FILTERS", font=("Arial", 12, "bold"), bg="#303134", fg="white")
        lbl_filt.pack(pady=(20, 10), anchor="w", padx=20)
        
        self.filter_var = tk.StringVar(value="Normal")
        filters = ["Normal", "Grayscale", "Sepia", "Invert", "Sketch", "Canny Edge"]
        
        for f in filters:
            rb = tk.Radiobutton(sidebar, text=f, variable=self.filter_var, value=f, command=self.set_filter,
                                bg="#303134", fg="white", selectcolor="#303134", activebackground="#303134", font=("Arial", 10))
            rb.pack(anchor="w", padx=30, pady=2)

        # --- AI Controls ---
        tk.Frame(sidebar, height=2, bg="grey").pack(fill=tk.X, pady=15, padx=10)
        lbl_ai = tk.Label(sidebar, text="ARTIFICIAL INTELLIGENCE", font=("Arial", 12, "bold"), bg="#303134", fg="white")
        lbl_ai.pack(anchor="w", padx=20)

        self.btn_ai = tk.Button(sidebar, text="ENABLE EMOTION AI", bg="#5f6368", fg="white", font=("Arial", 10, "bold"),
                                command=self.toggle_ai, width=20, pady=5)
        self.btn_ai.pack(pady=10)
        
        if self.emotion_model is None:
            self.btn_ai.config(state=tk.DISABLED, text="AI MODEL MISSING")

        # --- Snapshot ---
        tk.Frame(sidebar, height=2, bg="grey").pack(fill=tk.X, pady=15, padx=10)
        self.btn_snap = tk.Button(sidebar, text="📸 TAKE SNAPSHOT", bg=ACCENT_COLOR, fg="#202124", font=("Arial", 11, "bold"),
                                  command=self.take_snapshot, width=20, pady=8)
        self.btn_snap.pack(pady=10)

        # --- Exit ---
        btn_exit = tk.Button(sidebar, text="EXIT APP", bg="#e74c3c", fg="white", command=self.on_closing, width=20)
        btn_exit.pack(side=tk.BOTTOM, pady=20)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="System Ready")
        self.status_bar = tk.Label(self.window, textvariable=self.status_var, bg="#303134", fg="grey", anchor="w", padx=10)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def set_filter(self):
        self.current_filter = self.filter_var.get()

    def toggle_ai(self):
        self.detect_emotion = not self.detect_emotion
        if self.detect_emotion:
            self.btn_ai.config(bg=ACCENT_COLOR, fg="black", text="DISABLE AI")
            self.status_var.set("AI Engine: Active")
        else:
            self.btn_ai.config(bg="#5f6368", fg="white", text="ENABLE EMOTION AI")
            self.status_var.set("AI Engine: Standby")

    def apply_filter(self, frame):
        if self.current_filter == "Grayscale":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.current_filter == "Invert":
            return cv2.bitwise_not(frame)
        elif self.current_filter == "Canny Edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(gray, 50, 150)
        elif self.current_filter == "Sepia":
            img_sepia = np.array(frame, dtype=np.float64) # converting to float to prevent overflow
            img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                            [0.349, 0.686, 0.168],
                                                            [0.393, 0.769, 0.189]])) # Multiplying image with special sepia matrix
            img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
            return np.array(img_sepia, dtype=np.uint8)
        elif self.current_filter == "Sketch":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            inv_blur = cv2.bitwise_not(blur)
            return cv2.divide(gray, inv_blur, scale=256.0)
        return frame

    def process_ai(self, frame, gray_frame):
        # 1. Detect Faces
        faces = self.face_classifier.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        current_labels = []

        for i, (x, y, w, h) in enumerate(faces):
            label = ""
            
            # 2. Predict (Only every N frames to save CPU)
            if self.frame_count % self.process_every_n_frames == 0:
                try:
                    roi_gray = gray_frame[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray.astype("float") / 255.0
                    roi_gray = img_to_array(roi_gray)
                    roi_gray = np.expand_dims(roi_gray, axis=0)
                    
                    prediction = self.emotion_model.predict(roi_gray, verbose=0)[0]
                    label = self.emotion_labels[int(np.argmax(prediction))]
                except Exception:
                    label = "Unknown"
            else:
                # Use cached label if available
                if i < len(self.last_emotions):
                    label = self.last_emotions[i]
                else:
                    label = "Scanning..."
            
            current_labels.append(label)

            # Draw Box and Label
            color = (0, 255, 0) if label != "Angry" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.frame_count % self.process_every_n_frames == 0:
            self.last_emotions = current_labels

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            frame = cv2.flip(frame, 1) # Mirror
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            # Copy for AI (Clean input)
            ai_frame = frame.copy()
            
            # --- AI Layer ---
            if self.detect_emotion and self.emotion_model:
                gray = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2GRAY)
                self.process_ai(frame, gray) # Draws directly on 'frame'

            # --- Filter Layer ---
            display_frame = self.apply_filter(frame)
            
            # Show FPS
            if self.show_fps:
                cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- Tkinter Conversion ---
            # Handle grayscale filters
            if len(display_frame.shape) == 2:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)
            else:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(display_frame)
            # Resize to fit UI
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.window.after(10, self.update_frame)

    def take_snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = self.apply_filter(frame)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Snap_{ts}.png"
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Snapshot", f"Saved successfully:\n{filename}")

    def on_closing(self):
        self.is_running = False
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCameraApp(root, "Smart AI Camera v2.0")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
