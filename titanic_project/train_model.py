import joblib
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def fit_models(X_train, y_train):
    mlr = LogisticRegression()
    mlr.fit(X_train, y_train)

    des_tree = DecisionTreeClassifier()
    des_tree.fit(X_train, y_train)

    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)
    return mlr, des_tree, svm_model


def print_metrics(y_preds, y, comment: str):
    print("Metrics for", comment)
    print("Accuracy:", accuracy_score(y_preds, y))
    print("Classification Report:\n", classification_report(y_preds, y))


def main():
    features = ['Parch', 'SibSp', 'Pclass', 'Sex', 'Embarked', 'Age', 'Fare']
    target_col = ['Survived']
    df = pd.read_csv(input_file_name)
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, df[target_col], test_size=0.2)
    mlr_model, des_tree_model, svm_model_model = fit_models(X_train, y_train)
    models = [mlr_model, des_tree_model, svm_model_model]
    data = [X_train, X_test, y_train, y_test]
    joblib.dump((models, data), output_file_name)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 preprocess_data.py input_file_name output_file_name")
        sys.exit(1)

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    main()
