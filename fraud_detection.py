import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# para cargar el dataset
df = pd.read_csv('data/creditcard.csv')

# estopara ver las primeras filas para entender cómo son los datos
print(df.head())

# aqui la información básica del dataset
print(df.info())

# aqui comprobar si hay valores nulos en los datos
print(df.isnull().sum())

# ahora vamos a normalizar las características (escala de las variables numéricas)
scaler = StandardScaler()
df['NormalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# aqui vamos a eliminar la columna "Amount" original para no tener información redundante
df = df.drop(columns=['Amount'])

# vamos a dividir los datos en características (X) y la variable objetivo (y)
X = df.drop(columns=['Class'])  # Características
y = df['Class']  # Variable objetivo

#  aqui dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# vamos a crear el modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# ahora toca entrenar el modelo
rf_model.fit(X_train, y_train)

# vamos a realizar predicciones en el conjunto de prueba
y_pred = rf_model.predict(X_test)

# vamos a imprimir el reporte de clasificación
print(classification_report(y_test, y_pred))

# Mostrarndo la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ahora ya calcular y mostrar la AUC-ROC
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {roc_auc}")
