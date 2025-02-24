import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./data/DATASET_FINAL.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ada_boost = AdaBoostClassifier(
    n_estimators=50, 
    learning_rate=1.0, 
    random_state=None)

ada_boost.fit(X_train, y_train)
y_pred = ada_boost.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  
recall = recall_score(y_test, y_pred, average='weighted')  
f1 = f1_score(y_test, y_pred, average='weighted')  
roc_auc = roc_auc_score(y_test, ada_boost.predict_proba(X_test), multi_class='ovr') 

print(f'Exactitud: {accuracy:.4f}')
print(f'Precisión: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Área bajo la curva (AUC): {roc_auc:.4f}')
