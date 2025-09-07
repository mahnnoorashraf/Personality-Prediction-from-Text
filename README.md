# MBTI Personality Prediction from Text

A machine learning project that predicts Myers-Briggs Type Indicator (MBTI) personality types from written text using natural language processing and logistic regression.

## Overview

This project analyzes text data (such as social media posts, comments, or any written content) to predict one of the 16 MBTI personality types. The model uses TF-IDF vectorization and logistic regression to classify text into personality categories.

## Features

- **Text Preprocessing**: Advanced text cleaning including URL removal, lemmatization, and stop word filtering
- **TF-IDF Vectorization**: Converts text into numerical features for machine learning
- **Logistic Regression Model**: Multinomial classification for 16 MBTI types
- **Model Evaluation**: Comprehensive accuracy metrics and confusion matrix visualization
- **Interactive Prediction**: Function to predict MBTI type from any input text

## Dataset

- **Source**: MBTI dataset (`mbti_1.csv`)
- **Size**: 8,675 text posts
- **Features**: Text posts labeled with 16 MBTI personality types
- **MBTI Types**: All 16 combinations of:
  - **E/I**: Extraversion vs Introversion
  - **S/N**: Sensing vs Intuition  
  - **T/F**: Thinking vs Feeling
  - **J/P**: Judging vs Perceiving

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook (for running the .ipynb file)

### Required Libraries
```bash
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn
```

### Additional Setup
```python
# Download NLTK data
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Download spaCy model
import spacy
spacy.cli.download("en_core_web_sm")
```

## Usage

### Running the Notebook
1. Upload the `mbti_1.csv` dataset to your environment
2. Run all cells in `Code.ipynb`
3. The model will train automatically and display results

### Making Predictions
```python
# Example usage
sample_text = "I love spending time alone and reading books."
predicted_type = predict_mbti(sample_text)
print(f"Predicted MBTI: {predicted_type}")
```

## Model Performance

The model achieves classification accuracy on the test set with detailed metrics including:
- Overall accuracy score
- Per-class precision, recall, and F1-score
- Confusion matrix visualization

## Text Preprocessing Pipeline

1. **URL Removal**: Strips all URLs from text
2. **Character Filtering**: Removes special characters and numbers
3. **Lowercasing**: Converts all text to lowercase
4. **Tokenization**: Splits text into individual words
5. **Stop Word Removal**: Filters out common English stop words
6. **Lemmatization**: Reduces words to their root forms using spaCy

## Project Structure

```
personality prediction/
├── Code.ipynb          # Main Jupyter notebook with complete implementation
├── README.md           # This file
└── mbti_1.csv          # Dataset (upload required)
```

## Technical Details

- **Algorithm**: Logistic Regression with multinomial classification
- **Vectorization**: TF-IDF with 10,000 features
- **Train/Test Split**: 80/20 with stratified sampling
- **Solver**: LBFGS with 300 max iterations

## Example Predictions

| Input Text | Predicted MBTI |
|------------|----------------|
| "I love spending time alone and reading books." | INFJ (Introverted) |
| "I enjoy meeting new people and trying adventurous things!" | ENFP (Extraverted) |

## Limitations

- Model performance depends on the quality and representativeness of the training data
- Predictions are based on text patterns and may not capture the full complexity of personality
- Results should be interpreted as statistical predictions rather than definitive personality assessments

## Future Improvements

- Experiment with different machine learning algorithms (Random Forest, SVM, Neural Networks)
- Implement deep learning approaches using transformers
- Add feature engineering for better text representation
- Cross-validation for more robust model evaluation
- Real-time prediction API

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## License

This project is open source and available under the MIT License.
