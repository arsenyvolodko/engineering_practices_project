from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys


def write_metrics(y_preds, y, comment: str, file_name: str):
    with open(file_name, 'a') as f:
        f.write(f"Metrics for {comment}\n")
        f.write(f"Accuracy: {accuracy_score(y_preds, y)}\n")
        f.write(f"Classification Report:\n {classification_report(y_preds, y)}\n\n")


def main():
    try:
        models, data = joblib.load(input_file)
        X_train, X_test, y_train, y_test = data
        mlr_model, des_tree_model, svm_model = models
        models = [mlr_model, des_tree_model, svm_model]

        for model in models:
            write_metrics(model.predict(X_test), y_test, str(model), output_file)

    except Exception as e:
        print(f"Error loading data: {e}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 predict_model.py input_file output_file")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main()
