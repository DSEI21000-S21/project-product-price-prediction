import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def train_classification_model(model, X_train, X_test, y_train, y_test, target_classname):
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)


    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if len(y_train_pred.shape) > 1:
        y_train_pred = np.argmax(y_train_pred, axis=1)
        y_test_pred = np.argmax(y_test_pred, axis=1)


    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print()
    train_matrix = confusion_matrix(y_train, y_train_pred)
    train_class_acc = train_matrix.diagonal() / train_matrix.sum(axis=1)
    train_class_acc_indices = np.argsort(train_class_acc, )
    if len(train_class_acc_indices) > 10:
        print("Top 5 class with highest train accuracy")
        for i in train_class_acc_indices[::-1][:5]:
            print("%-20s - %.5f" % (target_classname[i], train_class_acc[i]))

        print("\nTop 5 class with lowest train accuracy")
        for i in train_class_acc_indices[:5]:
            print("%-20s - %.5f" % (target_classname[i], train_class_acc[i]))
    else:
        print(train_matrix)
        print("Train accuracy for each class")
        for i in train_class_acc_indices:
            print("%-20s - %.5f" % (target_classname[i], train_class_acc[i]))

    print("-" * 50)
    print("-" * 50)

    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print()
    test_matrix = confusion_matrix(y_test, y_test_pred)
    test_class_acc = test_matrix.diagonal() / test_matrix.sum(axis=1)
    test_class_acc_indices = np.argsort(test_class_acc, )
    if len(test_class_acc_indices) > 10:
        print("Top 5 class with highest test accuracy")
        for i in test_class_acc_indices[::-1][:5]:
            print("%-20s - %.5f"%(target_classname[i], test_class_acc[i]))

        print("\nTop 5 class with lowest test accuracy")
        for i in test_class_acc_indices[:5]:
            print("%-20s - %.5f" % (target_classname[i], test_class_acc[i]))
    else:
        print(test_matrix)
        print("Test accuracy for each class")
        for i in test_class_acc_indices:
            print("%-20s - %.5f" % (target_classname[i], test_class_acc[i]))


    return model, y_train, y_test, y_train_pred, y_test_pred
