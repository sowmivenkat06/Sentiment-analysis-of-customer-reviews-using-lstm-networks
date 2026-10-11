import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(
    page_title="SentimentAI - BiLSTM Review Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    .stAppHeader {
        background-color: transparent;
    }
    .sentiment-card {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .pos-card {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #10b981;
    }
    .neu-card {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.4);
        color: #f59e0b;
    }
    .neg-card {
        background: rgba(244, 63, 94, 0.15);
        border-color: rgba(244, 63, 94, 0.4);
        color: #f43f5e;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = 'best_bilstm_model.keras'
TOKENIZER_PATH = 'sentiment_tokenizer.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
MAX_LEN = 150

@st.cache_resource
def load_model_and_artifacts():
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, tokenizer, label_encoder

try:
    model, tokenizer, label_encoder = load_model_and_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header
st.title("🧠 SentimentAI Analyzer")
st.caption("Deep Learning BiLSTM Neural Network Sentiment Classification System")

st.sidebar.header("⚙️ Model Info")
st.sidebar.markdown(f"**Model:** BiLSTM Neural Network")
st.sidebar.markdown(f"**Sequence Length:** {MAX_LEN} tokens")
st.sidebar.markdown(f"**Classes:** {', '.join(label_encoder.classes_)}")
st.sidebar.markdown(f"**Vocab Size:** {len(tokenizer.word_index):,} words")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Review Input")
    
    user_text = st.text_area("Enter review:", height=150, placeholder="Type review here...")
    
    analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Sentiment Prediction")
    
    if analyze_clicked or user_text:
        if not user_text.strip():
            st.warning("Please enter a review text.")
        else:
            # Inference
            seqs = tokenizer.texts_to_sequences([user_text])
            padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
            probs = model.predict(padded)[0]
            
            best_idx = np.argmax(probs)
            label = label_encoder.classes_[best_idx]
            confidence = probs[best_idx] * 100
            
            # Display Banner
            card_class = "pos-card" if label == "Positive" else "neu-card" if label == "Neutral" else "neg-card"
            icon = "😊" if label == "Positive" else "😐" if label == "Neutral" else "😞"
            
            st.markdown(f"""
            <div class="sentiment-card {card_class}">
                <h2 style="margin:0;">{icon} Predicted Sentiment: <strong>{label}</strong></h2>
                <h3 style="margin:0; margin-top:8px;">Confidence: <strong>{confidence:.2f}%</strong></h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Class probabilities
            st.markdown("#### Class Probability Breakdown")
            for cls_name, prob in zip(label_encoder.classes_, probs):
                st.write(f"**{cls_name}**: {prob*100:.2f}%")
                st.progress(float(prob))

            # Internal Preprocessing Steps Expander
            cleaned_text = user_text.replace('!', ' exclamationmark ').replace('?', ' questionmark ').lower()
            cleaned_text = ' '.join(cleaned_text.split())
            token_seq = tokenizer.texts_to_sequences([cleaned_text])[0]

            with st.expander("🛠️ View Internal Preprocessing Steps"):
                st.markdown(f"**Cleaned Input Text:** `{cleaned_text}`")
                
                token_dict = {i: t for i, t in enumerate(token_seq[:20])}
                st.markdown("**Sequence Tokens (First 20):**")
                st.json(token_dict)
                
                st.markdown(f"**Total Words Tokenized:** `:green[{len(token_seq)}]`")
    else:
        st.info("Enter text on the left and click **Analyze Sentiment** to view predictions.")
