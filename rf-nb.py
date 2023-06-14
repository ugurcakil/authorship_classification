import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Step 1: Load the CSV file
data = pd.read_csv('/content/drive/MyDrive/out.csv')


X = data.drop('author', axis=1)  # Features
y = data['author']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Step 5: Create the Naive Bayes and Random Forest models
nb_model = MultinomialNB()
rf_model = RandomForestClassifier()

# Step 6: Create the Voting Classifier
voting_model = VotingClassifier(estimators=[('nb', nb_model), ('rf', rf_model)], voting='soft')

# Step 7: Train the Voting Classifier
voting_model.fit(X_train, y_train)

# Step 8: Make predictions
predictions = voting_model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
confusion_mat = confusion_matrix(y_test, predictions)

# Print accuracy, precision, recall, and F1 score
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# Plotting the confusion matrix
labels = sorted(set(y))
sns.heatmap(confusion_mat, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()