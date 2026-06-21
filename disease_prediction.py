import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Column names for the dataset
columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]

# Load dataset
data = pd.read_csv('diabetes.csv', names=columns)

print("Dataset Loaded Successfully")
print(data.head())

# Features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nDisease Prediction Accuracy:", accuracy)

# Example prediction
sample_patient = [[2, 120, 70, 30, 0, 25.0, 0.5, 25]]
result = model.predict(sample_patient)

if result[0] == 1:
    print("Prediction: Diabetic")
else:
    print("Prediction: Non-Diabetic")
