import os
import urllib.request

def create_dummy_model():
    """
    Creates a simple, untrained Keras model structure and saves it as emotion_model.h5
    This ensures the app runs. It will make random predictions until you replace it with a real model.
    """
    print("Generating placeholder AI model...")
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, Flatten, Dense
        
        # Simple structure matching common FER input (48x48 grayscale)
        model = Sequential([
            Conv2D(32, (3,3), input_shape=(48,48,1), activation='relu'),
            Flatten(),
            Dense(7, activation='softmax') # 7 emotions
        ])
        
        model.save('emotion_model.h5')
        print("✅ 'emotion_model.h5' created! (Note: This is an untrained dummy model)")
    except ImportError:
        print("❌ TensorFlow not installed. Skipping model generation.")

def download_haarcascade():
    filename = "haarcascade_frontalface_default.xml"
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"✅ {filename} downloaded.")
        except Exception as e:
            print(f"❌ Failed to download XML: {e}")
    else:
        print(f"✅ {filename} already exists.")

if __name__ == "__main__":
    download_haarcascade()
    create_dummy_model()
    print("\nSetup complete. You can now run 'main.py'.")
