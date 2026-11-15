# train_nn.py - train a small neural network + a logistic regression baseline,
# report accuracy, confusion matrix and calibration error (ECE).
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

SEED = 42

def ece(probs, y_true, classes, bins=10):
    """Expected calibration error: how far confidence is from real accuracy."""
    conf = probs.max(axis=1)
    pred = classes[probs.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total

def report(name, model, x_val, y_val, x_test, y_test):
    classes = model.classes_
    for split, x, y in (("val", x_val, y_val), ("test", x_test, y_test)):
        probs = model.predict_proba(x)
        acc = (classes[probs.argmax(axis=1)] == y).mean()
        print(f"{name:8s} {split:4s} accuracy={acc:.4f}  ECE={ece(probs, y, classes):.4f}")
    pred = model.predict(x_test)
    print(f"{name} confusion matrix (test):")
    print(confusion_matrix(y_test, pred))

def main():
    d = np.load("data/mnist_1to9.npz")
    x_train, y_train = d["x_train"], d["y_train"]
    x_val, y_val = d["x_val"], d["y_val"]
    x_test, y_test = d["x_test"], d["y_test"]

    nn = MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, random_state=SEED)
    nn.fit(x_train, y_train)

    lr = LogisticRegression(max_iter=200)
    lr.fit(x_train, y_train)

    report("NN", nn, x_val, y_val, x_test, y_test)
    report("LogReg", lr, x_val, y_val, x_test, y_test)

    joblib.dump(nn, "data/nn.joblib")
    joblib.dump(lr, "data/logreg.joblib")

if __name__ == "__main__":
    main()