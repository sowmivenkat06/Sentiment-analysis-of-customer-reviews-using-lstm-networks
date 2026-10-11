import os
import pickle
# pyrefly: ignore [missing-import]
import numpy as np
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, request, jsonify

# Suppress TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='static', template_folder='templates')

MODEL_PATH = 'best_bilstm_model.keras'
TOKENIZER_PATH = 'sentiment_tokenizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
MAX_LEN = 150

print("Loading Sentiment Analysis Model & Artifacts...")

# Load Tokenizer
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Load Label Encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Load Trained Keras Model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model and Tokenizer successfully loaded!")

def preprocess_text(text):
    cleaned = text.replace('!', ' exclamationmark ').replace('?', ' questionmark ').lower()
    cleaned = ' '.join(cleaned.split())
    return cleaned

def predict_sentiment_text(text):
    if not text or not text.strip():
        return None
    
    cleaned_text = preprocess_text(text)
    sequence_tokens = tokenizer.texts_to_sequences([cleaned_text])[0]
    padded = pad_sequences([sequence_tokens], maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict
    probabilities = model.predict(padded)[0]
    best_idx = int(np.argmax(probabilities))
    
    class_labels = list(label_encoder.classes_)
    predicted_label = class_labels[best_idx]
    confidence = float(probabilities[best_idx]) * 100
    
    breakdown = {}
    for i, label in enumerate(class_labels):
        breakdown[label] = round(float(probabilities[i]) * 100, 2)
        
    return {
        "sentiment": predicted_label,
        "confidence": round(confidence, 2),
        "breakdown": breakdown,
        "raw_probabilities": [float(p) for p in probabilities],
        "preprocessing": {
            "cleaned_text": cleaned_text,
            "sequence_tokens": sequence_tokens,
            "tokens_first_20": sequence_tokens[:20],
            "total_words_tokenized": len(sequence_tokens)
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Please enter a review text to analyze.'}), 400
        
    result = predict_sentiment_text(text)
    return jsonify(result)

@app.route('/api/model-info', methods=['GET'])
def model_info():
    vocab_size = len(tokenizer.word_index) if hasattr(tokenizer, 'word_index') else 'N/A'
    return jsonify({
        "model_name": "BiLSTM Sentiment Classifier",
        "input_sequence_length": MAX_LEN,
        "classes": list(label_encoder.classes_),
        "vocabulary_size": vocab_size,
        "padding_type": "post"
    })

if __name__ == '__main__':
    print("Starting Sentiment Analysis Web Server on http://localhost:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
