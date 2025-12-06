"""
TextMind: Personality Predictor - Model Training Module
Trains a Logistic Regression model to predict MBTI personality types from text.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re
import warnings

warnings.filterwarnings('ignore')


def preprocess_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def create_dummy_dataset():
    """
    Create a dummy dataset for demonstration purposes.
    
    Returns:
        pd.DataFrame: DataFrame with 'type' and 'posts' columns
    """
    dummy_data = [
        ("ESFP", "I love parties and meeting new people! Energy and excitement is what I live for. Can't wait for the weekend!"),
        ("INTJ", "I prefer reading alone and analyzing complex problems. Social gatherings drain my energy. Logic over emotions."),
        ("ENFP", "Life is an adventure! I jump from one idea to another. Spontaneity and creativity define me!"),
        ("ISTJ", "Structure and responsibility matter most. I follow rules and complete tasks on time. Reliability is key."),
        ("ENFJ", "People are my passion. I love helping others and bringing communities together. Harmony and connection matter."),
        ("INTP", "Deep dives into theory and abstract concepts fascinate me. I'd rather debate ideas than attend social events."),
        ("ESFJ", "I'm here to help and support my friends. Loyalty and tradition are important to me. Let's make everyone happy!"),
        ("ISFP", "Art and aesthetics speak to my soul. I appreciate beauty in small moments. Living in the present feels right."),
        ("ENTJ", "I lead by vision and strategy. Efficiency and achievement drive me. Emotions are secondary to results."),
        ("ISFJ", "I care deeply about people's feelings. Protecting those I love is my purpose. Quiet dedication defines me.")
    ]
    
    df = pd.DataFrame(dummy_data, columns=['type', 'posts'])
    return df


def train_model():
    """
    Main training function.
    Loads data (real or dummy), preprocesses it, and trains the model.
    """
    csv_file = 'mbti_1.csv'
    
    # Load or create dataset
    if os.path.exists(csv_file):
        print(f"✅ Found {csv_file}. Loading data...")
        df = pd.read_csv(csv_file)
        print(f"   Loaded {len(df)} records from CSV")
    else:
        print(f"⚠️  Warning: {csv_file} not found!")
        print("   Creating dummy dataset with 10 examples for demonstration...")
        df = create_dummy_dataset()
        print(f"   Dummy dataset created with {len(df)} records")
    
    print("\n📊 Dataset Info:")
    print(f"   Total records: {len(df)}")
    print(f"   Unique personality types: {df['type'].nunique()}")
    print(f"   Personality types: {sorted(df['type'].unique())}")
    
    # Preprocess text
    print("\n🧹 Preprocessing text...")
    df['posts_cleaned'] = df['posts'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['posts_cleaned'].str.len() > 0]
    print(f"   Texts after cleaning: {len(df)}")
    
    # Vectorize text using TF-IDF
    print("\n🔢 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['posts_cleaned'])
    
    print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"   Feature matrix shape: {X.shape}")
    
    # Encode labels
    print("\n🏷️  Encoding personality type labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['type'])
    
    print(f"   Classes: {label_encoder.classes_}")
    
    # Train Logistic Regression model
    print("\n🚀 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    model.fit(X, y)
    
    # Calculate training accuracy
    train_accuracy = model.score(X, y)
    print(f"   Training accuracy: {train_accuracy:.4f}")
    
    # Save model and vectorizer
    print("\n💾 Saving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   ✅ Saved model.pkl")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("   ✅ Saved vectorizer.pkl")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("   ✅ Saved label_encoder.pkl")
    
    print("\n✨ Training complete! Model is ready for inference.")


if __name__ == '__main__':
    train_model()
