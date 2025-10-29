import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
attrition = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\CSV\Employee Attrition.csv")

# Convert categorical column 'OverTime' into numeric (Yes=1, No=0)
attrition['OverTime'] = attrition['OverTime'].map({'Yes': 1, 'No': 0})

# Define only the 3 features used in API
X = attrition[['Age', 'MonthlyIncome', 'OverTime']]
y = attrition['Attrition'].map({'Yes': 1, 'No': 0})  # Assuming target column is 'Attrition'

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "attrition_model.pkl")

print("✅ New simplified attrition_model.pkl saved successfully!")
