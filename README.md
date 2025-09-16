# Simply example of Logistic Regression implementation 

This project implements Logistic Regression for binary classification, specifically tailored for classifying Amazon reviews as positive or negative.

## Features
- Training using stochastic gradient descent (SGD)
- Sigmoid function for probability estimation
- L2 regularization to prevent overfitting

## 1. Project Structure

```
logistic-regression/
├── dmia/
│   ├── __init__.py
│   ├── gradient_check.py
│   ├── utils.py
│   └── classifiers/
│       ├── __init__.py
│       └── logistic_regression.py
├── data/
│   └── train.csv
├── homework.ipynb
└── pyproject.toml
```

## 2. Logistic Regression Implementation

Please refer to the full `logistic_regression.py` in `dmia/classifiers/` for all implementation details including training, prediction, and loss functions.

More information can be found in the `homework.ipynb` notebook.

## 3. Testing Model

Create `test_solution.py`:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dmia.classifiers import LogisticRegression

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.train(X_train, y_train, learning_rate=0.1, num_iters=1000, batch_size=100, verbose=True)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.3f}")
```

## 4. Running

```powershell
jupyter notebook homework.ipynb
```

## Notes
1. Vectorized code for efficiency (no Python loops on big data operations).
2. Regularization does NOT apply to bias term.
3. Uses epsilon to avoid log(0) numerical issues.
4. Uses `scipy.sparse` for efficient sparse operations with text features.

