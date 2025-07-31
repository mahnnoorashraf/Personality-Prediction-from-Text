# 🧠 Personality Prediction from Text using MBTI Types

This project predicts a person's **MBTI (Myers-Briggs Type Indicator)** personality type based on their written text using machine learning and natural language processing.

---

## 📂 Dataset

- **Name**: MBTI 500
- **Source**: [Kaggle - MBTI 500](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **Format**: CSV with 2 columns:
  - `type`: MBTI personality label (e.g., INTJ, ENFP)
  - `posts`: A large block of user posts (text)

---

## 🚀 Project Workflow

1. **Data Loading**
   - Load the MBTI dataset using Pandas.

2. **Preprocessing**
   - Remove URLs, punctuation, special characters
   - Convert to lowercase
   - Tokenize and remove stopwords

3. **Feature Extraction**
   - Apply TF-IDF Vectorization to convert text into numerical vectors.

4. **Model Building**
   - Use Logistic Regression for multi-class classification.

5. **Evaluation**
   - Use accuracy score, classification report, and confusion matrix to evaluate the model.

---

## 📊 Technologies Used

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – data handling
  - `nltk` – natural language preprocessing
  - `scikit-learn` – ML model and evaluation
  - `matplotlib`, `seaborn` – data visualization

---

## 🧪 Example Prediction

```python
Input:  "I love working alone and thinking deeply about problems."
Output: "INTP"
