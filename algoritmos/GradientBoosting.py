import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('mantenibilidad', axis=1)
    y = df['mantenibilidad']
    
    categorical_columns = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for column in categorical_columns:
        X[column] = le.fit_transform(X[column])
    if y.dtype == 'object':
        y = le.fit_transform(y) 
    
    return X, y

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(
        loss='log_loss', 
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=42,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=0.0001,
        ccp_alpha=0.0
    )

    num_classes = len(np.unique(y))
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if num_classes == 2 else model.predict_proba(X_test)
    
    metrics = {
        'Exactitud': accuracy_score(y_test, y_pred),
        'Precisión': precision_score(y_test, y_pred, average='weighted' if num_classes > 2 else 'binary'),
        'Recall': recall_score(y_test, y_pred, average='weighted' if num_classes > 2 else 'binary'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted' if num_classes > 2 else 'binary'),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba, multi_class='ovr' if num_classes > 2 else 'raise')
    }
    
    return model, metrics, X_test, y_test

def plot_metrics(metrics):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title('Métricas de Evaluación del Modelo')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, X):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

def main(file_path):
    X, y = load_and_prepare_data(file_path)
    model, metrics, X_test, y_test = train_and_evaluate_model(X, y)
    
    print("\nMétricas de evaluación:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    plot_metrics(metrics)
    plot_feature_importance(model, X)
    
    return model, metrics

if __name__ == "__main__":
    file_path = "./data/DATASET_FINAL.csv"  
    model, metrics = main(file_path)
