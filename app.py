from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
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

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            emotion_label = detect_emotion(frame)
            yield f"data: {emotion_label}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='text/event-stream')

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    # Get image data from the request
    image_data = request.json.get('image_data')
    
    # Decode base64 image and convert to NumPy array
    decoded_image = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(decoded_image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform your emotion prediction here using a pre-trained model
    emotion_label = detect_emotion(img)
    
    return jsonify({'emotion_label': emotion_label})

if __name__ == '__main__':
    app.run(debug=True)
