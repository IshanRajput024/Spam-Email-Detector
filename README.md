# 📧 Spam Detection Using Naive Bayes

This project focuses on classifying emails as **spam** or **ham** (not spam) using a **Multinomial Naive Bayes** model. The model processes textual data and evaluates its performance through metrics like accuracy, precision, recall, and F1-score. 🚀  

## 🔑 Key Features
- **TF-IDF Vectorization**: Converts email content into numerical features for machine learning.
- **Naive Bayes Classifier**: Efficient and lightweight model for text classification.
- **Performance Metrics**: Includes accuracy, confusion matrix, and a detailed classification report.  

## 🛠️ Technologies Used
- **Python**: Core programming language.
- **Pandas**: Data manipulation.
- **scikit-learn**: For TF-IDF vectorization and machine learning algorithms.
- **NumPy**: For numerical operations.

## 🚀 How It Works
1. **Data Preprocessing**:
   - Splits the dataset into training (80%) and testing (20%) sets.
   - Vectorizes text using **TF-IDF** (Term Frequency-Inverse Document Frequency).
2. **Model Training**:
   - Trains a **Multinomial Naive Bayes** classifier on the vectorized training data.
3. **Evaluation**:
   - Measures model performance using accuracy, confusion matrix, and classification report.

## 📊 Evaluation Metrics
- **Accuracy**: How well the model correctly classifies emails.
- **Precision, Recall, F1-Score**: Measures for individual classes (spam and ham).
- **Confusion Matrix**: Displays correct and incorrect predictions in matrix form.

## 📂 Repository Structure
- `spam.csv`: The dataset containing email content and labels.
- `spam_detection.py`: The Python script for spam classification.
- `README.md`: Documentation and project overview.

## 💻 How to Run the Project
1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/spam-detection.git
