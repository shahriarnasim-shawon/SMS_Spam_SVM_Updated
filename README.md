# 📱 SMS Spam Classification using SVM

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)

## 📌 Project Overview
This repository contains a Machine Learning notebook that classifies SMS messages as either **Spam** (unwanted messages) or **Ham** (normal messages). The project relies on Natural Language Processing (NLP) techniques for text cleaning and a Support Vector Machine (SVM) for classification.

This project was built and updated in **Google Colab** using a Jupyter Notebook (`.ipynb`).

## 📊 Dataset
- **Source:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Description:** A set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged according being ham (legitimate) or spam.

## ✨ Key Features & Updates
The original base code has been updated with several best-practice machine learning enhancements:
- **Text Preprocessing:** Handled URLs, emails, numbers, and punctuation using Regex. Applied NLTK's `WordNetLemmatizer` and removed English stopwords.
- **Class Imbalance Handling:** Implemented `class_weight='balanced'` in the LinearSVC model to prevent bias toward the majority class ("Ham").
- **Hyperparameter Tuning:** Utilized `GridSearchCV` to find the optimal `C` parameter for the SVM pipeline.
- **Advanced Visualizations:** 
  - WordClouds for both Spam and Ham text bodies.
  - Plotted a **Confusion Matrix** to visualize True/False Positives and Negatives.
  - Plotted an **ROC Curve** (Receiver Operating Characteristic) to display the model's diagnostic ability and AUC score.
- **Model Persistence:** Saved the best performing pipeline using `joblib` (`sms_spam_model.pkl`) so the model can be deployed without needing to retrain.

## 🛠️ Technologies & Libraries Used
- **Data Manipulation:** `pandas`, `numpy`
- **NLP & Text Processing:** `nltk`, `re`, `string`, `WordCloud`
- **Machine Learning:** `scikit-learn` (LinearSVC, TfidfVectorizer, Pipeline, GridSearchCV)
- **Data Visualization:** `matplotlib`, `seaborn`
- **Model Saving:** `joblib`

## 🚀 How to Run the Notebook

### Option 1: Run in Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Click on **File > Open Notebook > GitHub** and paste the link to this repository.
3. Download the `spam.csv` dataset from the Kaggle link above.
4. Run the first cell to upload `spam.csv` into the Colab environment.
5. Click **Runtime > Run all** to execute the notebook.

### Option 2: Run Locally
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
Navigate to the project directory:
code
Bash
cd YOUR-REPOSITORY-NAME
Install the required dependencies:
code
Bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn joblib
Place the downloaded spam.csv file in the same directory.
Open the notebook using Jupyter:
code
Bash
jupyter notebook SMS_Spam_SVM_Updated.ipynb
📈 Results
The tuned SVM model achieved excellent results on the test set.
The Confusion Matrix shows how accurately the model separates Spam from Ham.
The ROC Curve demonstrates a high AUC (Area Under the Curve) score, proving the model is highly capable of distinguishing between the two classes.
✍️ Custom Predictions
The notebook includes a custom prediction function. You can pass any text to it to see how the model classifies it:
code
Python
predict_sms("Congratulations! You have won a free iPhone. Click here now.")
# Output: Spam 🚨

predict_sms("Hi, are we meeting tomorrow at 3 PM?")
# Output: Ham ✉️
