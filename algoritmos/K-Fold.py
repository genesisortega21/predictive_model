import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

df = pd.read_csv('./data/DATASET_FINAL.csv')

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values   

modelo = joblib.load("./modelo/modelo_mantenibilidad.pkl") 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(modelo, X, y, cv=cv, method="predict")

if len(set(y)) == 2:
    y_prob = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, y_prob)
else:
    y_bin = label_binarize(y, classes=np.unique(y))  
    y_prob = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")
    auc = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average="weighted")
recall = recall_score(y, y_pred, average="weighted")
f1 = f1_score(y, y_pred, average="weighted")

print("Resultados de Validación Cruzada con 5 Fold:")
print(f"Exactitud: {accuracy:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Área bajo la curva (AUC-ROC): {auc:.4f}")
