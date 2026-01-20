# 🎓 Student Pass/Fail Prediction

A machine learning project that predicts whether a student **passes or fails** based on academic and personal attributes using classification models.

## 📊 Dataset
- **File:** `student-mat.csv`
- **Target Variable:**
  - `1` → Pass (G3 ≥ 10)
  - `0` → Fail (G3 < 10)

## ⚙️ Preprocessing
- One-hot encoding applied to categorical features
- One category per feature dropped to prevent multicollinearity
- Feature scaling applied for Logistic Regression

## 🤖 Models Implemented
- Logistic Regression
- Decision Tree Classifier

## 📈 Evaluation
- Train-test split: 80% training / 20% testing
- Metric used: **Accuracy**

## 🛠️ Technologies Used
- Python
- pandas
- scikit-learn
- matplotlib

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/student-pass-fail-prediction.git
   cd student-pass-fail-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

3. **Run the script**
   ```bash
   python main.py
   ```

4. **View results**
   - Accuracy scores for Logistic Regression and Decision Tree models will be printed in the console.

## 🚀 Future Improvements
- Add cross-validation
- Tune hyperparameters
- Include additional evaluation metrics (Precision, Recall, F1-score)
