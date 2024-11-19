# MultiModal-Cyberbullying-Detection
This project utilizes machine learning techniques to detect cyberbullying in both textual and multimodal (image and audio) data. The goal is to create a system that can process and classify text extracted from various sources such as images (via OCR) and audio (via speech-to-text), using a deep learning model based on CNN-GRU architecture.

---

## Dataset

### `HateSpeechDatasetBalanced.csv`

This dataset contains labeled textual data related to cyberbullying. It consists of two columns:

- **Content**: The text content to be classified.
- **Label**: The label indicating whether the content is bullying (1) or not bullying (0).

The dataset is balanced to ensure equal representation of bullying and non-bullying content.

### Dataset Usage

The dataset is loaded from the CSV file and preprocessed by cleaning the text (removing punctuation, stopwords, and lemmatizing words). The cleaned data is then split into training and testing datasets. The text data is tokenized and padded before being fed into the CNN-GRU model for training and prediction.

---

## Model

### Model Architecture: CNN-GRU with Attention

The model combines Convolutional Neural Networks (CNN) and Gated Recurrent Units (GRU), with an attention mechanism and a global average pooling layer. It is designed to classify text data as either cyberbullying or non-cyberbullying.

- **Input**: Cleaned and tokenized text data.
- **Output**: Binary classification (cyberbullying detected or not).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score.

---

## Requirements

To run this project locally, make sure you have the following dependencies installed:

- Python 3.6+
- Flask
- TensorFlow
- NLTK
- OpenCV
- PaddleOCR
- SpeechRecognition
- sounddevice
- scipy

You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```
File Structure
The structure of the project is as follows:

plaintext
Copy code
.
├── templates/
│   └── index.html                # HTML template for the web app
├── app.py                         # Flask application file
├── cyberbullying_cnn_gru_model.h5  # Trained model
├── cyberbullyingdetection.ipynb    # Jupyter notebook for model training
├── HateSpeechDatasetBalanced.csv   # Dataset for training and evaluation
├── temp_audio.wav                 # Temporary audio file for testing
└── requirements.txt               # List of Python dependencies
How to Execute the Project
1. Clone the repository
Clone the repository to your local machine:

bash
Copy code
git clone <repository_url>
cd <project_directory>
2. Install dependencies
Install the required Python packages by running the following command:

bash
Copy code
pip install -r requirements.txt
3. Train the model (if necessary)
If the model is not already trained, you can train it using the Jupyter notebook (cyberbullyingdetection.ipynb):

Open cyberbullyingdetection.ipynb in Jupyter Notebook or Google Colab.
Follow the steps to preprocess the data, build the CNN-GRU model, train it, and save the model as cyberbullying_cnn_gru_model.h5.
4. Run the Flask app
Start the Flask web application by running:

bash
Copy code
python app.py
This will launch the web server on http://127.0.0.1:5001.

5. Interact with the Web App
Open your browser and go to http://127.0.0.1:5001. You will see a form where you can either:

Upload an image file (to classify text using OCR).
Record audio (to convert speech to text and classify).
The web app will display the result of the classification (whether the content is cyberbullying or not) along with the extracted text.

How the System Works
The system operates in three stages:

Image Classification:

Upload an image, which will be processed using PaddleOCR to extract text from the image.
The extracted text is then passed to the model for classification.
Audio Classification:

Record audio, which will be converted to text using the SpeechRecognition library.
The text is then classified as either cyberbullying or non-cyberbullying.
Text Classification:

You can also manually input text for classification.
The model will predict if the content is related to cyberbullying.
Evaluation Metrics
After training the model, the following metrics were evaluated on the test data:

Accuracy: 85.34%
Precision: 84.76%
Recall: 86.46%
F1 Score: 85.60%
