import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# LOAD DATASET
df = pd.read_csv("dataset/StudentsPerformance.csv")

# TARGET
y = df["math score"]

# PASS / FAIL
y = y.apply(lambda x: 1 if x >= 40 else 0)

# FEATURES
X = df.drop("math score", axis=1)

# CONVERT TEXT TO NUMBERS
X = pd.get_dummies(X)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
model = RandomForestClassifier()

# TRAIN MODEL
model.fit(X_train, y_train)

# SAVE MODEL
joblib.dump(model, "student_model.pkl")

print("MODEL TRAINED SUCCESSFULLY")