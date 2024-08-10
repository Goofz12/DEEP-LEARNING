import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# New dataset
data = {
    'House ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Size (sq ft)': [1500, 1800, 1700, 2200, 1200, 1600, 2000, 1400, 2100, 1300],
    'Bedrooms': [3, 4, 3, 4, 2, 3, 4, 3, 4, 2],
    'Bathrooms': [2, 3, 2, 3, 1, 2, 3, 2, 3, 1],
    'Sold': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Drop 'House ID' column
df = df.drop('House ID', axis=1)

# Encode the 'Sold' column
df['Sold'] = df['Sold'].apply(lambda x: 1 if x == 'Yes' else 0)

# Features and labels
X = df[['Size (sq ft)', 'Bedrooms', 'Bathrooms']]
y = df['Sold']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Print test data with predictions
df_test = X_test.copy()
df_test['Actual'] = y_test
df_test['Predicted'] = y_pred
df_test['Actual'] = df_test['Actual'].apply(lambda x: 'Yes' if x == 1 else 'No')
df_test['Predicted'] = df_test['Predicted'].apply(lambda x: 'Yes' if x == 1 else 'No')
print(df_test)

# Predict a new house
new_house = [[1900, 4, 3]]  # New house to be predicted
new_prediction = knn.predict(new_house)
new_prediction = 'Yes' if new_prediction[0] == 1 else 'No'
print(f'The new house is predicted to be sold: {new_prediction}')
