"""from flask import Flask, request, render_template
import cv2
import paddleocr
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess the text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Initialize the tokenizer (Use the same tokenizer as in training)
texts = ["sample text for tokenizer"]  # Replace with actual data
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

app = Flask(__name__)

# Initialize PaddleOCR Reader
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')  # PaddleOCR initialization

# Load the trained model
model = tf.keras.models.load_model('cyberbullying_ffnn.h5')

# Audio processing using sounddevice
def audio_to_text_using_sd():
    recognizer = sr.Recognizer()
    try:
        print("Recording audio...")
        duration = 10  # seconds
        sample_rate = 16000
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        audio_path = "temp_audio.wav"
        wav.write(audio_path, sample_rate, audio_data)
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        return None

# Classify extracted or input text
def classify_text(text):
    cleaned_text = clean_text(text)
    print(f"Cleaned Text: {cleaned_text}")  # Debugging step to check processed text
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=150, padding='post')
    
    if len(cleaned_text.split()) == 0:  # Check if the text is empty after cleaning
        return 'No valid text extracted', cleaned_text
    
    prediction = model.predict(padded_sequence)[0][0]
    result = 'Cyberbullying Detected' if np.round(prediction) == 1 else 'No Cyberbullying'
    return result, cleaned_text

# Image and text classification
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                # PaddleOCR text detection and recognition
                result = ocr.ocr(img, cls=True)
                extracted_text = " ".join([line[1][0] for line in result[0]])  # Extract recognized text
                print(f"Extracted Text: {extracted_text}")  # Debugging step to check OCR result
                prediction, cleaned_text = classify_text(extracted_text)
                return render_template("index.html", result=prediction, text=extracted_text, cleaned_text=cleaned_text)

        if "audio" in request.form:
            text = audio_to_text_using_sd()
            if text:
                prediction, cleaned_text = classify_text(text)
                return render_template("index.html", result=prediction, text=text, cleaned_text=cleaned_text)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5001)"""

from flask import Flask, request, render_template
import cv2
import paddleocr
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess the text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Initialize the tokenizer (Use the same tokenizer as in training)
texts = ["sample text for tokenizer"]  # Replace with actual data
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

app = Flask(__name__)

# Initialize PaddleOCR Reader
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')  # PaddleOCR initialization

# Load the trained model
model = tf.keras.models.load_model('cyberbullying_cnn_gru_model.h5')

# Audio processing using sounddevice
def audio_to_text_using_sd():
    recognizer = sr.Recognizer()
    try:
        print("Recording audio...")
        duration = 10  # seconds
        sample_rate = 16000
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        audio_path = "temp_audio.wav"
        wav.write(audio_path, sample_rate, audio_data)
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        return None

# Classify extracted or input text
def classify_text(text):
    print(f"Text for classification: {text}")  # Debugging step to check processed text
    sequence = tokenizer.texts_to_sequences([text])  # No cleaning, using raw extracted text
    padded_sequence = pad_sequences(sequence, maxlen=150, padding='post')
    
    if len(text.split()) == 0:  # Check if the text is empty
        return 'No valid text extracted', text
    
    prediction = model.predict(padded_sequence)[0][0]
    result = 'Cyberbullying Detected' if np.round(prediction) == 1 else 'No Cyberbullying'
    return result, text

# Image and text classification
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                # PaddleOCR text detection and recognition
                result = ocr.ocr(img, cls=True)
                extracted_text = " ".join([line[1][0] for line in result[0]])  # Extract recognized text
                print(f"Extracted Text: {extracted_text}")  # Debugging step to check OCR result
                prediction, _ = classify_text(extracted_text)
                return render_template("index.html", result=prediction, text=extracted_text)

        if "audio" in request.form:
            text = audio_to_text_using_sd()
            if text:
                prediction, _ = classify_text(text)
                return render_template("index.html", result=prediction, text=text)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5001)