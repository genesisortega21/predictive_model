import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv('./data/DATASET_FINAL.csv')

X = df.drop(columns=['mantenibilidad']) 
y = df['mantenibilidad']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42, dual=False)))  
]

final_estimator = LogisticRegression()
stacking_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)

if len(y.unique()) == 2:
    y_pred_prob = stacking_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
else:
    y_pred_prob = stacking_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  
recall = recall_score(y_test, y_pred, average='weighted')  
f1 = f1_score(y_test, y_pred, average='weighted')  

print(f'Exactitud: {accuracy:.4f}')
print(f'Precisión: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Área bajo la curva (AUC): {roc_auc:.4f}')
