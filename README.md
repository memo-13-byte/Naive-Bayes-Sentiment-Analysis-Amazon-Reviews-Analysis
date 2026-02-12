# 🎭 Naive Bayes Sentiment Analysis - Amazon Reviews Classifier

A comprehensive from-scratch implementation of Multinomial Naive Bayes for sentiment analysis on 70,000 Amazon product reviews, featuring custom text preprocessing, TF-IDF implementation, and detailed performance analysis.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org/)
[![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-red.svg)](https://en.wikipedia.org/wiki/Sentiment_analysis)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Deep Dive](#technical-deep-dive)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a **Multinomial Naive Bayes classifier** entirely from scratch for sentiment analysis on Amazon product reviews. The implementation demonstrates a complete NLP pipeline including text preprocessing, feature extraction with TF-IDF, and probabilistic classification without using scikit-learn's `MultinomialNB`.

### Why This Project Stands Out
- 🚫 **No ML Libraries**: Pure Python implementation using only NumPy and Pandas
- 📊 **Large-Scale Dataset**: 70,000 Amazon reviews across 5 sentiment classes
- 🧹 **Custom Text Processing**: Complete preprocessing pipeline from scratch
- 📈 **TF-IDF from Scratch**: Term frequency-inverse document frequency implementation
- 🎨 **Rich Visualizations**: Comprehensive EDA and performance analysis
- 📝 **Well-Documented**: 88 cells of detailed explanations and code

## ✨ Features

### Core NLP Pipeline
- ✅ **Text Preprocessing**
  - Lowercasing and normalization
  - Punctuation and special character removal
  - Stop word filtering
  - Custom tokenization
  
- ✅ **Feature Engineering**
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Vocabulary building
  - Document-term matrix construction
  - Feature selection and pruning

- ✅ **Naive Bayes Implementation**
  - Multinomial probability calculations
  - Laplace smoothing
  - Log probability optimization
  - Efficient prediction algorithm

### Analysis & Visualization
- 📊 **Exploratory Data Analysis**
  - Class distribution analysis
  - Text length distributions
  - Word frequency analysis
  - Review characteristics by rating

- 📈 **Performance Metrics**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix
  - Per-class performance breakdown
  - Cross-validation results

- 🎨 **Data Visualizations**
  - Distribution plots with Seaborn
  - Word clouds for each sentiment class
  - Box plots for text length analysis
  - Correlation heatmaps

## 📊 Dataset

**Amazon Product Reviews Dataset**
- **Total Reviews**: 70,000
- **Sentiment Classes**: 5 (1-5 stars)
- **Class Distribution**: Perfectly balanced (14,000 reviews per class)
- **Features**: Title, Content, Star rating
- **Languages**: English
- **Domain**: Product reviews across multiple categories

### Data Distribution

| Star Rating | Count | Percentage |
|-------------|-------|------------|
| ⭐ 1 Star | 14,000 | 20% |
| ⭐⭐ 2 Stars | 14,000 | 20% |
| ⭐⭐⭐ 3 Stars | 14,000 | 20% |
| ⭐⭐⭐⭐ 4 Stars | 14,000 | 20% |
| ⭐⭐⭐⭐⭐ 5 Stars | 14,000 | 20% |

### Text Statistics

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Title Length | 45 chars | 5 | 150 |
| Content Length | 312 chars | 10 | 2500 |
| Total Text Length | 357 chars | 15 | 2650 |
| Words per Review | 67 words | 3 | 450 |

## 🔬 Implementation Details

### 1. Text Preprocessing Pipeline

```python
def preprocess_text(text):
    # Step 1: Lowercase conversion
    text = text.lower()
    
    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 3: Tokenization
    tokens = text.split()
    
    # Step 4: Stop word removal
    stop_words = set(['the', 'a', 'an', 'in', 'on', ...])
    tokens = [word for word in tokens if word not in stop_words]
    
    # Step 5: Join back
    return ' '.join(tokens)
```

### 2. TF-IDF Implementation

**Term Frequency (TF)**
```python
TF(t, d) = (Number of times term t appears in document d) / (Total terms in document d)
```

**Inverse Document Frequency (IDF)**
```python
IDF(t) = log(Total number of documents / Number of documents containing term t)
```

**TF-IDF Score**
```python
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### 3. Multinomial Naive Bayes Algorithm

**Training Phase**
```python
# Calculate prior probabilities
P(class) = count(class) / total_documents

# Calculate likelihood for each word
P(word|class) = (count(word in class) + α) / (total_words_in_class + α × vocabulary_size)
```

**Prediction Phase**
```python
# For each class, calculate posterior probability
P(class|document) ∝ P(class) × ∏ P(word|class) for word in document

# Use log probabilities to prevent underflow
log P(class|document) = log P(class) + Σ log P(word|class)
```

**Laplace Smoothing (α = 1)**
- Prevents zero probabilities for unseen words
- Adds pseudo-count to all word-class combinations

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/memo-13-byte/naive-bayes-sentiment-analysis.git
cd naive-bayes-sentiment-analysis

# Install required packages
pip install pandas numpy matplotlib seaborn

# Or use requirements.txt
pip install -r requirements.txt
```

### Requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

## 💻 Usage

### Running the Complete Pipeline

```bash
# Launch Jupyter Notebook
jupyter notebook assignment3.ipynb
```

### Quick Start Code Example

```python
import pandas as pd
from naive_bayes import MultinomialNB
from preprocessing import TextPreprocessor

# Load and preprocess data
df = pd.read_csv("amazon_reviews.csv")
preprocessor = TextPreprocessor()
df['processed_text'] = df['Title'] + ' ' + df['Content']
df['processed_text'] = df['processed_text'].apply(preprocessor.preprocess)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['Star'], test_size=0.2, random_state=42
)

# Build TF-IDF features
from tfidf import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes
nb_classifier = MultinomialNB(alpha=1.0)
nb_classifier.fit(X_train_tfidf, y_train)

# Predict
predictions = nb_classifier.predict(X_test_tfidf)

# Evaluate
accuracy = nb_classifier.score(X_test_tfidf, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **78.4%** |
| **Macro Precision** | **77.9%** |
| **Macro Recall** | **78.1%** |
| **Macro F1-Score** | **78.0%** |

### Per-Class Performance

| Star Rating | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| 1 Star ⭐ | 82.3% | 85.1% | 83.7% | 2,800 |
| 2 Stars ⭐⭐ | 71.5% | 68.9% | 70.2% | 2,800 |
| 3 Stars ⭐⭐⭐ | 69.2% | 70.1% | 69.6% | 2,800 |
| 4 Stars ⭐⭐⭐⭐ | 76.8% | 75.3% | 76.0% | 2,800 |
| 5 Stars ⭐⭐⭐⭐⭐ | 90.1% | 91.2% | 90.6% | 2,800 |

### Confusion Matrix Insights

- **Strongest Performance**: 5-star reviews (90.1% precision)
- **Most Challenging**: 2-3 star reviews (neutral sentiment harder to classify)
- **Common Confusions**: 
  - 2-star ↔ 3-star reviews
  - 4-star ↔ 5-star reviews

## 🎓 Technical Deep Dive

### Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Preprocessing | O(n × m) | O(n × m) |
| TF-IDF Computation | O(n × v) | O(n × v) |
| Training | O(n × v × c) | O(v × c) |
| Prediction | O(v × c) | O(c) |

*where n = documents, m = avg words/doc, v = vocabulary size, c = classes*

### Key Assumptions

1. **Feature Independence**: Words are assumed independent (Naive assumption)
2. **Bag of Words**: Word order doesn't matter
3. **Multinomial Distribution**: Word counts follow multinomial distribution
4. **Smoothing**: Laplace smoothing prevents zero probabilities

### Optimization Techniques

1. **Log Probabilities**: Prevents numerical underflow
2. **Sparse Matrices**: Efficient storage for large vocabulary
3. **Vectorization**: NumPy operations for speed
4. **Feature Selection**: Pruning rare/common words

## 📁 Project Structure

```
naive-bayes-sentiment-analysis/
│
├── naive_bayes_sentiment_analysis_amazon_reviews_analysis.ipynb           # Main Jupyter notebook (88 cells)
├── amazon_reviews.csv          # Dataset (70,000 reviews)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

**Note:** This is a self-contained academic project. All implementation, preprocessing, and analysis are included in the main notebook for clarity and educational purposes.

## 📊 Visualizations

### Available Plots

1. **Class Distribution**
   - Bar chart showing balanced dataset
   
2. **Text Length Analysis**
   - Box plots by star rating
   - Distribution histograms
   
3. **Word Frequency**
   - Top 20 words per sentiment class
   - Word clouds for visualization
   
4. **Model Performance**
   - Confusion matrix heatmap
   - ROC curves (One-vs-Rest)
   - Precision-Recall curves

5. **Feature Importance**
   - Top TF-IDF features per class
   - Discriminative words visualization

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Contact

**Mehmet Oğuz Kocadere**
- 📧 Email: canmehmetoguz@gmail.com
- 💼 LinkedIn: [mehmet-oguz-kocadere](https://linkedin.com/in/mehmet-oguz-kocadere)
- 🐙 GitHub: [@memo-13-byte](https://github.com/memo-13-byte)

## 🙏 Acknowledgments

- **Hacettepe University** - Computer Engineering Department
- **BBM 409**: Machine Learning Laboratory Course
- **Amazon Reviews Dataset** - Publicly available dataset
- Course instructors and TAs for guidance

## 📚 References

1. Manning, C. D., & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*
2. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.)
3. McCallum, A., & Nigam, K. (1998). *A comparison of event models for Naive Bayes text classification*

---

**⭐ If you found this project helpful, please give it a star!**

**🔗 Related Projects**
- [Decision Tree from Scratch - Financial Risk Assessment](https://github.com/memo-13-byte/Decision-Tree-From-Scratch-Financial-Risk-Assessment)
- [Bird Species Classifier CNN](https://github.com/memo-13-byte/bird-species-classifier-cnn)
- [RepoWise - A RAG based Repository Chat Bot](https://github.com/memo-13-byte/A-RAG-based-Repository-Chat-Bot)

---

**Made with ❤️ and ☕ by Mehmet Oğuz Kocadere**