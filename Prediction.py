<<<<<<< HEAD
# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading and reading the dataset
heart = pd.read_csv(r"D:\Projects\Heart Disease\heart_cleveland_upload.csv")

# Creating a copy of the dataset so that it will not affect our original dataset
heart_df = heart.copy()

# Renaming some of the columns
heart_df = heart_df.rename(columns={'condition': 'target'})
print(heart_df.head())

# Model building
# Fixing our data in X and y. Here y contains target data and X contains the rest of the features.
X = heart_df.drop(columns='target')
y = heart_df['target']

# Splitting our dataset into training and testing using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use transform here

# Creating Random Forest classifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = model.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-random-forest-model.pkl'  # Updated filename
=======
# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading and reading the dataset
heart = pd.read_csv(r"D:\Projects\Heart Disease\heart_cleveland_upload.csv")

# Creating a copy of the dataset so that it will not affect our original dataset
heart_df = heart.copy()

# Renaming some of the columns
heart_df = heart_df.rename(columns={'condition': 'target'})
print(heart_df.head())

# Model building
# Fixing our data in X and y. Here y contains target data and X contains the rest of the features.
X = heart_df.drop(columns='target')
y = heart_df['target']

# Splitting our dataset into training and testing using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use transform here

# Creating Random Forest classifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train_scaled, y_train)

# Making predictions
y_pred = model.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-random-forest-model.pkl'  # Updated filename
>>>>>>> 4f2b27e41e6a2f8ad1c64224f8c0dd219a9d86b2
pickle.dump(model, open(filename, 'wb'))