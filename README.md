## ✨ Features

### 🧠 Artificial Intelligence
*   **Real-time Face Detection:** Automatically tracks faces in the video feed.
*   **Emotion Recognition:** Classifies facial expressions into 7 categories:
    *   Angry 😡
    *   Disgust 🤢
    *   Fear 😱
    *   Happy 😄
    *   Sad 😢
    *   Surprise 😲
    *   Neutral 😐

### 🎨 Visual Filters
*   **Normal:** Standard high-definition video feed.
*   **Grayscale:** Classic black and white processing.
*   **Sepia:** Vintage/Retro photo effect.
*   **Invert:** Negative color effect.
*   **Sketch:** Real-time pencil drawing simulation.
*   **Canny Edge:** Highlights object outlines and edges.

### ⚙️ Application Capabilities
*   **Performance Optimized:** Implements **Frame Skipping** (AI processes every 5th frame) to ensure smooth FPS on standard laptops.
*   **Snapshot Mode:** Capture and save processed images instantly with a timestamp.
*   **Modern GUI:** Dark-themed interface built with Tkinter for a professional look.
*   **Robust Error Handling:** The app runs even if AI models are missing (features gracefully disable).

## 📂 Project Structure

```text
├── main.py              # The core application source code
├── setup_models.py      # Helper script to generate/download required files
├── emotion_model.h5     # The AI Model (Weights file)
├── haarcascade...xml    # OpenCV Face Detection classifier
└── README.md            # Project documentation
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-ai-camera.git
cd smart-ai-camera
```

### 2. Install Dependencies
Ensure you have Python installed. Run the following command to install the required libraries:
```bash
pip install opencv-python tensorflow pillow numpy
```

### 3. Initialize Files
Run the setup script to automatically download the Face Detection XML and generate a placeholder AI model file:
```bash
python setup_models.py
```

### 4. Run the App
```bash
python main.py
```

## ⚠️ Important Note Regarding the AI Model

The `setup_models.py` script creates a **dummy** `emotion_model.h5` file so that the application runs immediately without errors. However, this dummy model is untrained and will output random predictions.

**To get accurate Emotion Detection:**
1.  Download a pre-trained **FER-2013** Keras model (you can find these on GitHub or Kaggle).
2.  Rename the downloaded file to `emotion_model.h5`.
3.  Replace the file in your project directory.
4.  Restart the application.

## 🕹️ Controls

*   **Filter Panel:** Use the radio buttons on the right sidebar to switch visual effects instantly.
*   **Enable Emotion AI:** Click this button to toggle the TensorFlow engine. It will start drawing boxes around faces and predicting emotions.
*   **Take Snapshot:** Click to save the current frame (with filters applied) to the project folder.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/NewFilter`).
3.  Commit your changes (`git commit -m 'Add NewFilter'`).
4.  Push to the branch (`git push origin feature/NewFilter`).
5.  Open a Pull Request.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
