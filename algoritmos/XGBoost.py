import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import numpy as np

data = pd.read_csv('./data/DATASET_FINAL.csv')

X = data.drop('mantenibilidad', axis=1)  
y = data['mantenibilidad']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    random_state=None,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='auc',  
    early_stopping_rounds=10
)

num_classes = len(np.unique(y))

if num_classes == 2:
    model.set_params(eval_metric='auc')  
    auc_func = roc_auc_score
else:
    model.set_params(eval_metric='merror')
    auc_func = lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovr')

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted' if num_classes > 2 else 'binary')
recall = recall_score(y_test, y_pred, average='weighted' if num_classes > 2 else 'binary')
f1 = f1_score(y_test, y_pred, average='weighted' if num_classes > 2 else 'binary')
auc = auc_func(y_test, model.predict_proba(X_test)[:, 1] if num_classes == 2 else model.predict_proba(X_test))

print(f'Exactitud: {accuracy:.4f}')
print(f'Precisión: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Área bajo la curva (AUC): {auc:.4f}')
