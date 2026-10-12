# Synthetic customer churn example for learning KNN and SVM.
# The generated data is artificial and does not represent real-world churn performance.

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create synthetic binary classification data
X, y = make_classification(
    n_samples=500,
    n_features=4,
    n_classes=2,
    random_state=42,
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Scale the features using only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Train SVM model
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions
knn_pred = knn.predict(X_test_scaled)
svm_pred = svm.predict(X_test_scaled)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Print results in percentage format
print("KNN Accuracy:", f"{knn_accuracy * 100:.2f}%")
print("SVM Accuracy:", f"{svm_accuracy * 100:.2f}%")

if knn_accuracy > svm_accuracy:
    print("Comparison: KNN has higher test accuracy than SVM.")
elif svm_accuracy > knn_accuracy:
    print("Comparison: SVM has higher test accuracy than KNN.")
else:
    print("Comparison: KNN and SVM have tied test accuracy.")
