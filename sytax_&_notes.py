# Part 1: Review and Select a Dataset
# Binary Classification of the Customer Churn Dataset



# Part 2: Preprocess the Data
# Import the necessary libraries
#
# import pandas as pd
#   from sklearn.model_selection import train_test_split
#   from sklearn.linear_model import LogisticRegression
#   from sklearn.preprocessing import StandardScaler
#   from sklearn.neighbors import KNeighborsClassifier
#   from sklearn.ensemble import RandomForestClassifier
#   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
#
#   data = pd.read_csv('customer_churn.csv')
#   data.head()
#
# Separeate Fetures and Target
#
#   X = data.drop('Churn', axis=1)
#   y = data['Churn']
#   X.head()
#   y.head()
#
# Perform the train-test split
#
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# Diplay the shapes of the resulting datasets
#
#   print(X_train.shape)
#   print(X_test.shape)
#   print(y_train.shape)
#   print(y_test.shape)



# Part 3: Train and Evaluate the Models
# Logistic Regression
#
# Prepare the dataset
#
#   display(X_train, X_test, y_train, y_test)
#
# Scale the Features
#
#   scaler = StandardScaler()
#
# Fit the scaler on the TRAINING DATA and transform it
#
#   X_train = scaler.fit_transform(X_train)
#
# Use the same scaler to transform the TEST DATA
#
#   X_test_scaled = scaler.transform(X_test)
#
# Initialize the Logistic Regression Model
#
#   logistic_model = LogisticRegression(random_state=42)
#   logistic_model.fit(X_train_scaled, y_train)
#
# Evaluate the model
# Make predictions on the scaled test data
#
#   y_pred = logistic_model.predict(X_test_scaled)
#
# Evaluate the model
#
#   log_accuracy = accuracy_score(y_test, y_pred)
#   log_conf_matrix = confusion_matrix(y_test, y_pred)
#   log_class_report = classification_report(y_test, y_pred)
#
# Print the evaluation metrics
#
#   print(f"Logistic Regression Accuracy: {log_accuracy}")
#   print("Confusion Matrix:")
#   print(log_conf_matrix)
#   print("Classification Report:")
#   print(log_class_report)

# K-Nearest Neighbors
#
# Prepare and scale the dataset
#
#   scaler = StandardScaler()
#
# Fit the scaler on the training data and transform it
#
#   X_train_scaled = scaler.fit_transform(X_train)
#
# Use the same scaler to transform the test data
#
#   X_test_scaled = scaler.transform(X_test)
#
# Train the KNN model
# Initialize the KNN model
#
#   knn_model = KNeighborsClassifier(n_neighbors=5)
#
# Train the model on the scaled training data
#
#   knn_model.fit(X_train_scaled, y_train)
#
# Evaluate the model
# Make predictions on the scaled test data
#
#   y_pred = knn_model.predict(X_test_scaled)
#
# Evaluate the model
#
#   k_accuracy = accuracy_score(y_test, y_pred)
#   k_conf_matrix = confusion_matrix(y_test, y_pred)
#   k_class_report = classification_report(y_test, y_pred)
#
# Print the evaluation metrics
#
#   print(f"K-Nearest Neighbors Accuracy: {k_accuracy}")
#   print("Confusion Matrix:")
#   print(k_conf_matrix)
#   print("Classification Report:")
#   print(k_class_report)
#
# Random Forest Classifier
# Prepare and scale the dataset
#
#   scaler = StandardScaler()
#
# Fit the scaler on the training data and transform it
#
#   X_train_scaled = scaler.fit_transform(X_train)
#
# Use the same scaler to transform the test data
#
#   X_test_scaled = scaler.transform(X_test)
#
# Train the Random Forest Classifier
# Initialize the Random Forest Classifier
#
#   random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
#
# Train the model on the scaled training data
#
#   random_forest_model.fit(X_train_scaled, y_train)
#
# Evaluate the model
# Make predictions on the scaled test data
#
#   y_pred = random_forest_model.predict(X_test_scaled)
#
# Evaluate the model
#
#   ran_accuracy = accuracy_score(y_test, y_pred)
#   ran_conf_matrix = confusion_matrix(y_test, y_pred)
#   ran_class_report = classification_report(y_test, y_pred)
#
# Print the evaluation metrics
#
#   print(f"Random Forest Classifier Accuracy: {ran_accuracy}")
#   print("Confusion Matrix:")
#   print(ran_conf_matrix)
#   print("Classification Report")
#   print(ran_class_report)
#
#   print(f"     Logistic Regression Accuracy:  {format(log_accuracy * 100, '.2f')}%")
#   print(f"                     KNN Accuracy:  {format(k_accuracy * 100, '.2f')}%")
#   print(f"Random Forest Classifier Accuracy:  {format(ran_accuracy * 100, '.2f')}%")


# Part 4: Conclusion: Select the Best Model