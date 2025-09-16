# test_solution.py
from scipy import sparse  # ensure inputs are sparse CSR
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dmia.classifiers import LogisticRegression

# Create synthetic dataset (dense by default)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Convert to CSR sparse matrix so append_biases (sparse.hstack) receives sparse input
X = sparse.csr_matrix(X)  # works with sklearn train_test_split; outputs stay CSR when input is sparse

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # accepts scipy.sparse inputs

# Train our implementation
clf = LogisticRegression()
clf.train(X_train, y_train, learning_rate=0.1, num_iters=1000, batch_size=100, verbose=True)  # sparse-safe path

# Make predictions
y_pred = clf.predict(X_test)  # classifier handles bias internally; test passes sparse matrix through

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)  # standard sklearn metric
print(f"Test accuracy: {accuracy:.3f}")
