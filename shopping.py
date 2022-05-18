import csv
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

    # Confusion matrix
    classifier1 = KNeighborsClassifier(n_neighbors=1)
    classifier1.fit(X_train, y_train)

    classifier3 = KNeighborsClassifier(n_neighbors=3)
    classifier3.fit(X_train, y_train)

    y_pred1 = classifier1.predict(X_test)
    print("\nThe confusion matrix for k = 1:")
    print(confusion_matrix(y_test, y_pred1))
    print ("Accuracy : ", accuracy_score(y_test, y_pred1))

    y_pred3 = classifier3.predict(X_test)
    print("\nThe confusion matrix for k = 3:")
    print(confusion_matrix(y_test, y_pred3))
    print ("Accuracy : ", accuracy_score(y_test, y_pred3))


def load_data(filename):

    df = pd.read_csv(filename, delimiter =',')

    month_map = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6,
                "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}

    df['Month'] = df['Month'].map(month_map)
    df["VisitorType"] = df["VisitorType"].map(lambda x : 1 if x == "Returning_Visitor" else 0)
    df["Weekend"] = df["Weekend"].map(lambda x : 1 if x else 0)
    df["Revenue"] = df["Revenue"].map(lambda x : 1 if x else 0)

    int_map = ["Administrative", "Informational", "ProductRelated", "Month", "OperatingSystems", "Browser",
               "Region", "TrafficType", "VisitorType", "Weekend"]
    
    float_map = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates",
                 "ExitRates", "PageValues", "SpecialDay"]

    for type in int_map:
        if df[type].dtype != "int64":
            df = df.astype({type : "int64"})

    for type in float_map:
        if df[type].dtype != "float64":
            df = df.astype({type : "float64"})

    evidence = df.iloc[:,:-1].values.tolist()
    labels = df.iloc[:,-1].values.tolist()

    return (evidence, labels)

    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    #raise NotImplementedError


def train_model(evidence, labels):

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    #raise NotImplementedError


def evaluate(labels, predictions):

    tn = 0
    tp = 0
    fn = 0
    fp = 0

    for label, pred in zip(labels, predictions):
        if label == 1:
            if pred == 1:
                tp += 1
            else:
                fn += 1
        
        else:
            if pred == 1:
                fp += 1
            else:
                tn += 1

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return (sensitivity, specificity)

    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    #raise NotImplementedError


if __name__ == "__main__":
    main()
