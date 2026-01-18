# 🤖 AI Echo: Your Smartest Conversational Partner

AI Echo is an end-to-end **NLP-based Sentiment Analysis project** that analyzes user reviews of a ChatGPT-style application and classifies them into **Positive, Neutral, and Negative** sentiments.  
The project converts unstructured customer feedback into actionable business insights using **Machine Learning and Streamlit**.

---

## 📌 Domain
**Customer Experience & Business Analytics**

---

## 🎯 Problem Statement
User reviews are typically unstructured text data, making manual analysis inefficient and error-prone.  
The goal of this project is to apply **Natural Language Processing (NLP)** techniques to automatically analyze and classify user reviews based on sentiment, helping businesses understand customer satisfaction and identify improvement areas.

---

## 🧠 Business Use Cases
- Customer Feedback Analysis  
- Brand Reputation Monitoring  
- Feature Enhancement Decisions  
- Automated Complaint Prioritization  
- Marketing Strategy Optimization  

---

## 📊 Dataset Description
Dataset: `chatgpt_style_reviews_dataset.xlsx`

Key columns:
- `date` – Review submission date  
- `review` – User feedback text  
- `rating` – Rating from 1 to 5  
- `platform` – Web or Mobile  
- `location` – User country  
- `version` – ChatGPT version  
- `verified_purchase` – Verified user or not  

---

## ⚙️ Data Preprocessing
- Converted text to lowercase  
- Removed punctuation, numbers, and special characters  
- Removed stopwords  
- Applied **lemmatization** to preserve semantic meaning  
- Handled missing values  
- Derived sentiment labels using rating logic:
  - 4–5 → Positive  
  - 3 → Neutral  
  - 1–2 → Negative  

---

## 📈 Exploratory Data Analysis (EDA)
- Rating distribution analysis  
- Sentiment distribution visualization  
- Platform-wise and version-wise rating comparison  
- Time-series analysis of ratings  
- Word clouds for positive and negative reviews  

---

## 🤖 Machine Learning Approach
- **Feature Engineering:** TF-IDF Vectorization  
- **Model Used:** Logistic Regression  
- **Why Logistic Regression?**
  - Performs well on high-dimensional sparse text data  
  - Fast, efficient, and interpretable  

---

## 📏 Model Evaluation
The model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

F1-score was prioritized due to class imbalance.

---

## 🌐 Streamlit Dashboard
The trained model is deployed using **Streamlit** to provide an interactive dashboard.

### Dashboard Features:
- Sentiment distribution (pie chart)  
- Rating distribution (bar chart)  
- Platform & version filters  
- Word clouds for positive and negative reviews  
- Time-series rating trend  
- Real-time sentiment prediction for new reviews  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NLTK, Scikit-learn, Matplotlib, WordCloud  
- **NLP:** TF-IDF, Lemmatization  
- **Deployment:** Streamlit  

---

## 📁 Project Structure
AI_Echo/
├── app.py
├── data/
│ ├── raw/chatgpt_style_reviews_dataset.xlsx
│ └── processed/cleaned_reviews.csv
├── models/
│ ├── sentiment_model.pkl
│ └── vectorizer.pkl
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_model_training.ipynb
├── requirements.txt
└── README.md


---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

---

### 2️⃣Run the Streamlit app

streamlit run app.py

### 🚀 Future Enhancements

Use deep learning models like LSTM or BERT

Multilingual sentiment analysis

Real-time data ingestion via APIs

Cloud deployment (AWS / Streamlit Cloud)

### ✅ Conclusion

AI Echo demonstrates a complete Data Science lifecycle, from data preprocessing and NLP to machine learning modeling and deployment.
The project provides valuable insights into customer sentiment and can be scaled for real-world applications.

## 👩‍💻 Author

Revathy
Aspiring Data Scientist

