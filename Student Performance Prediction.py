# Goal : weather it is pass or fail.
# Key Features : Study Hours , Attendance, Previous Scores

# Model : Logistic Regression , Decision Tree Classifier


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('student-mat.csv', sep=';')
df = pd.DataFrame(data)


df = pd.get_dummies(df, columns=['school','sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'], drop_first=True)


# -------------------------
# Separate features & target
# -------------------------

x = df.drop(columns=['G3'])
y = (df['G3'] >= 10).astype(int)  # Target value: 1 for pass (G3 >= 10), 0 for fail


# # For New Locan Data Prediction : We will use only key features
# x = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
# y = (df['G3'] >= 10).astype(int)  # Target value: 1 for pass (G3 >= 10), 0 for fail


# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -------------------------
# Feature scaling (for Logistic Regression)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ----- Logistic Regression -----
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", log_model.score(X_test_scaled, y_test))

# ----- Decision Tree Classifier -----
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree Classifier Accuracy:", dt_model.score(X_test, y_test))
# -------------------------


# -------------------------
# Plotting Models to compare accuracy
# ------------------------- 

models = ['Logistic Regression', 'Decision Tree Classifier']
accuracies = [] 
# ----- Logistic Regression -----
accuracies.append(log_model.score(X_test_scaled, y_test))
# ----- Decision Tree Classifier -----
accuracies.append(dt_model.score(X_test, y_test))
# Plotting
plt.bar(models, accuracies, color=['blue', 'green'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

# Test with new data

# new_data = {
#     'studytime': 2,
#     'failures': 0,
#     'absences': 4,
#     'G1': 12,
#     'G2': 14 
# }
# new_df = pd.DataFrame([new_data])
# new_df_scaled = scaler.transform(new_df)
# log_new_pred = log_model.predict(new_df_scaled)
# dt_new_pred = dt_model.predict(new_df)
# print("Logistic Regression Prediction for new data (1=Pass, 0=Fail):", log_new_pred[0])
# print("Decision Tree Classifier Prediction for new data (1=Pass, 0=Fail):", dt_new_pred[0])





