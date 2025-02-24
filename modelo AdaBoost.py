import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=["mantenibilidad"])
    y = df["mantenibilidad"]
    return X, y

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=None)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {"accuracy": accuracy, "classification_report": report}
    
    return model, metrics, X_test, y_test

def save_model(model, filename):
    joblib.dump(model, filename)

def main(file_path):
    X, y = load_and_prepare_data(file_path)
    model, metrics, X_test, y_test = train_and_evaluate_model(X, y)
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    save_model(model, 'modelo_mantenibilidad.pkl')
    
    return model, metrics

if __name__ == "__main__":
    file_path = "./data/DATASET_FINAL.csv"
    model, metrics = main(file_path)
