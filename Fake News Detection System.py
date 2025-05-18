# Fake News Detection System with Streamlit and Neural Network
# A comprehensive solution for detecting fake news using NLP and deep learning

import numpy as np
import pandas as pd
import re
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class FakeNewsDetector:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.vectorizer = None
        self.model_type = None

    def load_data(self, filepath):
        """Load the dataset from the given filepath"""
        try:
            data = pd.read_csv(filepath)
            st.success(f"Data loaded successfully with shape: {data.shape}")
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def preprocess_text(self, text):
        """Preprocess text by removing special characters, stemming, and lemmatization"""
        if isinstance(text, float):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def prepare_data(self, data, text_column, label_column):
        """Prepare data for training by preprocessing text"""
        df = data.copy()
        df[text_column] = df[text_column].fillna('')
        st.info("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        X = df['processed_text']
        y = df[label_column].map({'REAL': 0, 'FAKE': 1})  # Convert labels to numeric
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, model_type='logistic'):
        """Train a selected model on the preprocessed data"""
        self.model_type = model_type
        st.info(f"Training {model_type} model...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)

        if model_type == 'logistic':
            self.model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000)
            self.model.fit(X_train_vec, y_train)
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=10, random_state=42)
            self.model.fit(X_train_vec, y_train)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_vec, y_train)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_vec, y_train)
        elif model_type == 'neural_network':
            self.model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train_vec.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            X_train_dense = X_train_vec.toarray()
            self.model.fit(X_train_dense, y_train, epochs=10, batch_size=32, verbose=0)
        else:
            raise ValueError("Model type not recognized")
        st.success("Model training completed.")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model on test data"""
        if self.model is None:
            st.error("Model not trained yet. Please train the model first.")
            return None, None, None
        X_test_vec = self.vectorizer.transform(X_test)
        if self.model_type == 'neural_network':
            X_test_vec = X_test_vec.toarray()
            y_pred = (self.model.predict(X_test_vec) > 0.5).astype(int).flatten()
            y_pred_proba = self.model.predict(X_test_vec)
        else:
            y_pred = self.model.predict(X_test_vec)
            y_pred_proba = self.model.predict_proba(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
        st.write(f"**Model Accuracy**: {accuracy:.4f}")
        st.write("**Confusion Matrix**:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        st.write("**Classification Report**:")
        st.text(report)
        return accuracy, conf_matrix, report

    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model is None:
            st.error("Model not trained yet. Please train the model first.")
            return False
        try:
            if self.model_type == 'neural_network':
                self.model.save(filepath)
            else:
                joblib.dump({'model': self.model, 'vectorizer': self.vectorizer}, filepath)
            st.success(f"Model saved to {filepath}")
            return True
        except Exception as e:
            st.error(f"Error saving model: {e}")
            return False

    def load_model(self, filepath, model_type):
        """Load a previously trained model from disk"""
        self.model_type = model_type
        try:
            if model_type == 'neural_network':
                self.model = tf.keras.models.load_model(filepath)
                self.vectorizer = joblib.load('vectorizer.pkl')  # Vectorizer saved separately
            else:
                loaded = joblib.load(filepath)
                self.model = loaded['model']
                self.vectorizer = loaded['vectorizer']
            st.success(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def predict(self, text):
        """Predict whether the given text is fake or real news"""
        if self.model is None or self.vectorizer is None:
            st.error("Model or vectorizer not loaded. Please train or load a model first.")
            return None
        processed_text = self.preprocess_text(text)
        X_vec = self.vectorizer.transform([processed_text])
        if self.model_type == 'neural_network':
            X_vec = X_vec.toarray()
            prediction = (self.model.predict(X_vec) > 0.5).astype(int)[0]
            probability = self.model.predict(X_vec)[0][0]
        else:
            prediction = self.model.predict(X_vec)[0]
            probability = np.max(self.model.predict_proba(X_vec))
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': float(probability)
        }
        return result

# Streamlit UI
def main():
    st.set_page_config(page_title="Fake News Detector", layout="wide")
    
    # Custom CSS for transparent, glassmorphism UI
    st.markdown("""
        <style>
        .main {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 20px;
        }
        .stButton>button {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background: rgba(255, 255, 255, 0.4);
        }
        .stTextInput>div>input {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📰 Fake News Detector")
    st.markdown("A tool to detect fake news using NLP and machine learning techniques.")

    detector = FakeNewsDetector()

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Train Model", "Predict", "Load Model"])

    if page == "Home":
        st.header("Welcome to Fake News Detector")
        st.write("""
            This application allows you to:
            - Upload a dataset and train various models (including a neural network) to detect fake news.
            - Make predictions on new text inputs.
            - Load pre-trained models for predictions.
            - Visualize model performance with confusion matrices and classification reports.
        """)

    elif page == "Train Model":
        st.header("Train a Model")
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
        model_type = st.selectbox("Select Model Type", 
                                 ["logistic", "decision_tree", "random_forest", "gradient_boosting", "neural_network"])
        text_column = st.text_input("Text Column Name", value="text")
        label_column = st.text_input("Label Column Name", value="label")
        
        if st.button("Train Model"):
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                X_train, X_test, y_train, y_test = detector.prepare_data(data, text_column, label_column)
                detector.train_model(X_train, y_train, model_type)
                detector.evaluate_model(X_test, y_test)
                detector.save_model(f"fake_news_detector_{model_type}.pkl")
            else:
                st.error("Please upload a dataset.")

    elif page == "Predict":
        st.header("Make a Prediction")
        sample_text = st.text_area("Enter text to classify", 
                                  "Enter news article text here...")
        if st.button("Predict"):
            if sample_text:
                prediction = detector.predict(sample_text)
                if prediction:
                    st.write(f"**Prediction**: {prediction['prediction']} (Confidence: {prediction['confidence']:.2f})")
            else:
                st.error("No model loaded. Please train or load a model first.")

    elif page == "Load Model":
        st.header("Load a Pre-trained Model")
        model_type = st.selectbox("Select Model Type", 
                                 ["logistic", "decision_tree", "random_forest", "gradient_boosting", "neural_network"])
        uploaded_model = st.file_uploader("Upload your model file (.pkl for non-neural, .h5 for neural network)", type=["pkl", "h5"])
        if st.button("Load Model"):
            if uploaded_model is not None:
                detector.load_model(uploaded_model.name, model_type)
            else:
                st.error("Please upload a model file.")

if __name__ == "__main__":
    main()