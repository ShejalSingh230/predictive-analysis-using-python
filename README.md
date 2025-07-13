# 📊 Customer Churn Prediction using Random Forest

This project performs customer churn prediction using a **Random Forest Classifier**. The dataset contains customer data from a telecom company. The goal is to predict whether a customer will **Stay** or **Churn**, based on features like age, tenure, services used, billing information, and more.

---

## 📁 Project Structure

```
📦Customer-Churn-Prediction
 ┣ 📄 Customer Prediction Data.xlsx         # Excel file with training and new data
 ┣ 📄 churn_prediction.py                   # Python script for model training & prediction
 ┣ 📄 Predictions.csv                       # Output file with churned customer predictions
 ┗ 📄 README.md                             # Documentation
```

---

## 🔧 Technologies Used

- 🐍 Python 3.x  
- 📊 Pandas, NumPy  
- 📈 Matplotlib, Seaborn  
- 🤖 Scikit-learn  
- 💾 Joblib  
- 📘 openpyxl (for Excel handling)

---

## 📌 Dataset Overview

### 📑 Sheet 1: `vw_ChurnData`
- Training dataset
- Target column: `Customer_Status` → `Stayed` (0), `Churned` (1)

### 📑 Sheet 2: `vw_JoinData`
- New customers for prediction
- Model will predict if they are likely to churn

---

## ✅ Steps Performed

### 1️⃣ Import Libraries & Load Data

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
```

---

### 2️⃣ Data Preprocessing

- Dropped irrelevant columns (e.g. `Customer_ID`, `Gender`, etc.)
- Label encoded categorical features
- Converted target variable `Customer_Status` to binary (0 = Stayed, 1 = Churned)

---

### 3️⃣ Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 4️⃣ Model Training

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

---

### 5️⃣ Evaluation

```python
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

📌 **Model Accuracy**: `83%`  
📊 **Precision (Churned)**: `77%`  
📉 **Recall (Churned)**: `59%`  
🧠 **F1 Score (Churned)**: `67%`

---

### 6️⃣ Feature Importance

```python
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=X.columns[indices])
```

Top important features:
- `Tenure_in_Months`
- `Monthly_Charge`
- `Internet_Service`

---

### 7️⃣ Prediction on New Data

- Loaded `vw_JoinData`
- Preprocessed similarly as training data
- Predicted churn status
- Saved predicted churners to `Predictions.csv`

---

## 💡 How to Run the Code

1. 🔽 Clone the Repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. ⚙️ Install Dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl joblib
```

3. 🚀 Run the Script:
```bash
python churn_prediction.py
```

---

## 📝 Output

🗂️ `Predictions.csv` → Contains customers predicted to **Churn**

---

## 👩‍💻 Author

**Shejal Sanjeev Singh**  
💼 Data Analyst | Python Developer  
📧 singhshejal899@gmail.com 

---

## 📜 License

This project is licensed under the **MIT License** 📝
