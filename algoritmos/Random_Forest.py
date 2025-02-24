import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./data/DATASET_FINAL.csv')

X = df.drop(['mantenibilidad'], axis=1)
y = df['mantenibilidad']
X = X.apply(LabelEncoder().fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None, 
    min_samples_split=2,
    min_samples_leaf=1, 
    max_features='sqrt', 
    bootstrap=True, 
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  
recall = recall_score(y_test, y_pred, average='weighted')  
f1 = f1_score(y_test, y_pred, average='weighted') 

if len(y.unique()) == 2:
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
else:
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print(f'Exactitud: {accuracy:.4f}')
print(f'Precisión: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Área bajo la curva (AUC): {roc_auc:.4f}')
