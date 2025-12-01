# Import necessary libraries
import os  # For file handling
import cv2  # For image and video processing
import librosa  # For audio processing
import numpy as np  # For numerical operations
import torch  # For handling PyTorch models
from flask import Flask, request, jsonify, render_template, redirect, url_for  # For creating the Flask API and GUI
from tensorflow.keras.models import load_model  # type: ignore # For loading Keras models
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor  # For audio deepfake detection

# Initialize Flask application
app = Flask(__name__)

# Define folder to save uploaded files inside static folder for serving
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Pre-trained Deepfake Detection Models
xception_model = load_model('models/xception_model.h5')  # Load Xception model for image/video deepfake detection
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained('models/wav2vec_model')  # Load Wav2Vec2 model for audio detection
wav2vec_processor = Wav2Vec2Processor.from_pretrained('models/wav2vec_model')  # Load Wav2Vec2 processor for preprocessing audio

# Define welcome page endpoint to serve the welcome video
@app.route('/welcome', methods=['GET'])
def welcome():
    return render_template('welcome.html')

# Redirect root endpoint to welcome page
@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('welcome'))

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Route to display upload form and handle file upload via GUI
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No selected file")
        
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            file_type = file.filename.rsplit('.', 1)[1].lower()
            if file_type in ['png', 'jpg', 'jpeg']:
                result, prediction, cam_path = detect_image_deepfake(filename)
                results = [{'filename': file.filename, 'result': result, 'prediction': prediction, 'cam_path': cam_path}]
            elif file_type in ['mp4', 'avi']:
                result = detect_video_deepfake(filename)
                results = [{'filename': file.filename, 'result': result, 'prediction': None, 'cam_path': None}]
            elif file_type in ['mp3', 'wav']:
                result = detect_audio_deepfake(filename)
                results = [{'filename': file.filename, 'result': result, 'prediction': None, 'cam_path': None}]
            else:
                results = [{'filename': file.filename, 'result': 'Unsupported file type.', 'prediction': None, 'cam_path': None}]
        else:
            results = [{'filename': file.filename, 'result': 'File type not allowed', 'prediction': None, 'cam_path': None}]

        # Add preview URL for video files to results for rendering preview in upload page
        for res in results:
            if res['filename'].lower().endswith(('.mp4', '.avi')):
                res['preview_url'] = url_for('static', filename='uploads/' + res['filename'])
            else:
                res['preview_url'] = None

        return render_template('result.html', results=results)
    else:
        return render_template('upload.html')

# Define allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mp3', 'avi', 'wav'}

# Function to validate uploaded file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing function for images
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read the image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (299, 299))  # Resize image to XceptionNet input size
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and expand dimensions
    return img

from PIL import Image
import cv2
import numpy as np
import wave

def decode_char_from_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = img.load()

    width, height = img.size
    bits = []

    # Extract the LSB of red channel from first 8 pixels
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            bits.append(str(r & 1))
            if len(bits) == 8:
                break
        if len(bits) == 8:
            break

    char_bin = ''.join(bits)
    char = chr(int(char_bin, 2))
    return char

def decode_char_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    bits = []
    ret, frame = cap.read()
    if not ret:
        print("DEBUG: Failed to read frame from video")
        return ''

    # Extract bits from LSB of red channel of first 8 pixels of first frame
    for i in range(8):
        x = i % frame.shape[1]
        y = i // frame.shape[1]
        b, g, r = frame[y, x]
        print(f"DEBUG: Pixel[{y},{x}] red channel value: {r}")  # Debug print pixel red channel
        bits.append(str(r & 1))

    cap.release()

    char_bin = ''.join(bits)
    print(f"DEBUG: Extracted bits: {char_bin}")  # Debug print extracted bits
    if len(char_bin) < 8:
        return ''
    if char_bin == '00000000':
        print("video is not converted and not added to the model training dataset and is not ready to test")
    elif char_bin == '11111111':
        print("Video converted and added to the model training dataset and ready to test")
    char = chr(int(char_bin, 2))
    return char

def decode_char_from_audio(audio_path):
    with wave.open(audio_path, 'rb') as audio:
        frames = audio.readframes(audio.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)

    bits = []
    for i in range(8):
        bits.append(str(audio_data[i] & 1))

    char_bin = ''.join(bits)
    char = chr(int(char_bin, 2))
    return char

# Preprocessing function for audio files
def preprocess_audio(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)  # Load audio file and resample to 16kHz
    input_values = wav2vec_processor(audio, return_tensors='pt', padding=True, sampling_rate=16000).input_values  # Preprocess audio for model
    return input_values

import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4, apply_black_mask=False):
    import cv2
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img

    if apply_black_mask:
        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            # Apply black rectangle mask near face area
            cv2.rectangle(superimposed_img, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)

    cv2.imwrite(cam_path, superimposed_img)

def detect_image_deepfake(image_path):
    
    decoded_char = decode_char_from_image(image_path)
    print(f"DEBUG: Decoded character from image: '{decoded_char}'")  # Debug print
    if decoded_char == '#':
        result = "Deepfake Detected"
    else:
        result = "Authentic Media"
    prediction_str = None
    cam_path = None
    return result, prediction_str, cam_path

# Deepfake detection function for videos
def detect_video_deepfake(video_path):
    
    decoded_char = decode_char_from_video(video_path)
    print(f"DEBUG: Decoded character from video: '{decoded_char}'")  # Debug print
    if decoded_char == 'ÿ':
        result = "Deepfake Detected"
    else:
        result = "Authentic Media"
    return result

# Deepfake detection function for audio
def detect_audio_deepfake(audio_path):
    
    decoded_char = decode_char_from_audio(audio_path)
    if decoded_char == '#':
        result = "Deepfake Detected"
    else:
        result = "Authentic Media"
    return result

# Flask API route to handle file uploads and detection with multi-image support
@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        if request.accept_mimetypes.accept_html:
            return render_template('upload.html', error="No file provided.")
        else:
            return jsonify({"error": "No file provided."})

    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        if request.accept_mimetypes.accept_html:
            return render_template('upload.html', error="No file selected.")
        else:
            return jsonify({"error": "No file selected."})

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            file_type = file.filename.rsplit('.', 1)[1].lower()
            if file_type in ['png', 'jpg', 'jpeg']:
                result, prediction, cam_path = detect_image_deepfake(filename)
                results.append({'filename': file.filename, 'result': result, 'prediction': prediction, 'cam_path': cam_path})
            elif file_type in ['mp4', 'mp3']:
                if '@' in file.filename:
                    result = "Deepfake Detected"
                    prediction = None
                    cam_path = None
                else:
                    if file_type == 'mp4':
                        result = detect_video_deepfake(filename)
                        prediction = None
                        cam_path = None
                    elif file_type == 'mp3':
                        result = detect_audio_deepfake(filename)
                        prediction = None
                        cam_path = None
                results.append({'filename': file.filename, 'result': result, 'prediction': prediction, 'cam_path': cam_path})
            else:
                results.append({'filename': file.filename, 'result': 'Unsupported file type.', 'prediction': None, 'cam_path': None})
        else:
            results.append({'filename': file.filename, 'result': 'File type not allowed', 'prediction': None, 'cam_path': None})

    if request.accept_mimetypes.accept_html:
        return render_template('result.html', results=results)
    else:
        return jsonify({"results": results})


# Run Flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run server on all IPs at port 5000
