import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def train_classification_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
    if len(y_train_pred.shape) > 1:
        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_test_pred = np.argmax(y_test_pred, axis=1)

    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print(confusion_matrix(y_train, y_train_pred))

    print("-"*50)
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))

    return model, y_train_pred, y_test_pred
