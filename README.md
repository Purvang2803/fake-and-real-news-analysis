```markdown
# 📰 Fake News Detection using Machine Learning and Deep Learning

This project focuses on detecting fake news articles using both traditional machine learning classifiers and deep learning models. It leverages a labeled dataset of real and fake news articles to train, evaluate, and compare the performance of multiple classification approaches.

## 📁 Dataset Description

The project uses two datasets:

- **Fake.csv** – Contains 23,481 fake news articles  
- **True.csv** – Contains 21,417 real news articles

Each dataset includes the following columns:

| Column Name | Description |
|-------------|-------------|
| `title`     | The headline of the news article |
| `text`      | The full content of the article |
| `subject`   | The topic or category (e.g., politics, news) |
| `date`      | Publication date |

The datasets were combined into a single dataframe and labeled for binary classification:  
- `0` → Fake news  
- `1` → Real news

## 🧠 Models Implemented

The following models were trained and evaluated:

1. Logistic Regression  
2. Multinomial Naive Bayes  
3. LSTM (Long Short-Term Memory)  
4. BiLSTM (Bidirectional LSTM)

### ⚙️ Feature Engineering

- **TF-IDF Vectorization** for traditional ML models  
- **Word2Vec Embeddings** for LSTM-based models  
- **Text cleaning**, stopword removal, and tokenization applied to prepare the data

## 📊 Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-Score

### 🏆 Best Performing Model

The **Bidirectional LSTM (BiLSTM)** achieved the best overall performance across all metrics. Its ability to capture both past and future context in text made it particularly effective in identifying fake news patterns.

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- NLTK, Gensim  
- Matplotlib, Seaborn

## 📈 Visualizations

The notebook includes:

- Confusion matrices  
- Metric comparison bar charts  
- Word clouds and top TF-IDF terms

## 📦 Project Structure

```
.
├── Fake.csv
├── True.csv
├── project_1.ipynb
├── README.md
└── requirements.txt (optional)
```

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Launch the Jupyter Notebook:

```bash
jupyter notebook project_1.ipynb
```

## 🔮 Future Improvements

- Integrate attention layers with BiLSTM  
- Explore transformer-based models like BERT  
- Deploy as a web app for real-time prediction
