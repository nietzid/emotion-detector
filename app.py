from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('fer2013_emotion_model.h5')

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    emotion_prediction = model.predict(reshaped)
    emotion_label = np.argmax(emotion_prediction)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotion_labels[emotion_label]

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            emotion_label = detect_emotion(frame)
            yield f"data:{emotion_label}\n\n"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
