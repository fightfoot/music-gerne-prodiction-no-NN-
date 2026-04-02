import numpy as np
import pandas as pd


class LogisticRegressionOVR:
    def __init__(self, a=0.01, epochs=1000):
        self.a = a  #  learning rate
        self.epochs = epochs
        self.weights = {}
        self.bias = {}
        self.classes = None
        self.mean_ = None
        self.std_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, w, b):
        """ Compute cost function """
        m = len(y)
        z = np.dot(X, w) + b
        h = self.sigmoid(z)
        h = np.clip(h, 1e-10, 1 - 1e-10)

        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def gradient_descent(self, X, y, w, b):
        """ Compute gradients for weights and bias """
        m = len(y)
        z = np.dot(X, w) + b
        h = self.sigmoid(z)

        dw = (1 / m) * np.dot(X.T, (h - y))
        db = (1 / m) * np.sum(h - y)

        return dw, db

    def fit_scaler(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def train_one_vs_rest(self, X, y):
        """ Train logistic regression for each class separately using One-vs-Rest """
        m, n = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            y_binary = (y == c).astype(int)
            w = np.zeros(n)
            b = 0.0

            for epoch in range(self.epochs):
                dw, db = self.gradient_descent(X, y_binary, w, b)

                w -= self.a * dw
                b -= self.a * db

                if epoch % 100 == 0:
                    cost = self.compute_cost(X, y_binary, w, b)
                    print(f"Class {c}, Epoch {epoch}: Cost = {cost:.4f}")

            self.weights[c] = w
            self.bias[c] = b

    def predict(self, X):
        """Compute class probabilities for all samples and return the class with highest probability."""
        class_scores = []

        for c in self.classes:
            z = np.dot(X, self.weights[c]) + self.bias[c]
            probs = self.sigmoid(z)
            class_scores.append(probs)

        class_scores = np.array(class_scores).T
        predicted_indices = np.argmax(class_scores, axis=1)
        return self.classes[predicted_indices]


file_path = "master_file.csv"
df = pd.read_csv(file_path)

df["artist_name_word_count"] = df["artist_name"].fillna("").apply(lambda x: len(str(x).split()))
df["title_word_count"] = df["title"].fillna("").apply(lambda x: len(str(x).split()))

feature_columns = [
    "duration",
    "tempo",
    "key",
    "mode",
    "loudness",
    "time_signature",
    "year",
    "artist_name_word_count",
    "title_word_count"
]

target_column = "genre"
split_column = "train_test"

df = df[df[split_column].isin(["TRAIN", "TEST"])].copy()

train_df = df[df[split_column] == "TRAIN"].copy()
test_df = df[df[split_column] == "TEST"].copy()

X_train = train_df[feature_columns].values.astype(float)
y_train = train_df[target_column].values
X_test = test_df[feature_columns].values.astype(float)
y_test = test_df[target_column].values

model = LogisticRegressionOVR(a=0.01, epochs=1000)  # a = Learning rate

model.fit_scaler(X_train)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

model.train_one_vs_rest(X_train, y_train)

y_pred = model.predict(X_test)
acc = np.mean(y_pred == y_test) * 100

print(f"\nTest Accuracy: {acc:.2f}%")