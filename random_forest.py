import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
data = pd.read_csv('/content/drive/MyDrive/out.csv')

# Step 2: Preprocess the data (if needed)

# Step 3: Split the data into training and testing sets
X = data.drop('author', axis=1)  # Features
y = data['author']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=84)



# Step 5: Create the Random Forest model
model = RandomForestClassifier()

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Make predictions
predictions = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
confusion_mat = confusion_matrix(y_test, predictions)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# Plotting the confusion matrix
labels = sorted(set(y))
sns.heatmap(confusion_mat, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()