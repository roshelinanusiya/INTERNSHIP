import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load the dataset
df = pd.read_csv('titanic.csv')

# 2. Preprocessing
# Drop columns that won't help in prediction
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Encode Categorical Data
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex']) # male=1, female=0

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# 3. Split data into Features (X) and Target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the model and encoders
# We save encoders so we can use the same mapping in our App
joblib.dump(model, 'titanic_model.joblib')
joblib.dump(le_sex, 'le_sex.joblib')
joblib.dump(le_embarked, 'le_embarked.joblib')

print("Model and encoders saved successfully!")
