

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the MNIST dataset
print("Loading and sampling MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)

# Convert data and labels
X, y = mnist.data, mnist.target.astype(int)

# Downsample to 10,000 samples for faster training
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=10000, stratify=y, random_state=42)

# Normalize the pixel values to [0, 1]
X_sample /= 255.0

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

# Train the SVM model
print("Training the SVM with linear kernel...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Step 1: Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract images (X) and labels (y)
X, y = mnist.data, mnist.target.astype(np.uint8)  # Convert labels to integers

# Step 2: Normalize the data (scale pixel values between 0 and 1)
X = X / 255.0

# Step 3: Split dataset into training (60,000 samples) and testing (10,000 samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=56000, random_state=42)

# Step 4: Train the SVM classifier
svm_model = SVC(kernel='rbf', C=10, gamma=0.01)  # RBF kernel works well for digit classification
svm_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = svm_model.predict(X_test)

# Step 6: Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", class_report)

# Step 7: Plot the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for MNIST SVM Classifier")
plt.show()
