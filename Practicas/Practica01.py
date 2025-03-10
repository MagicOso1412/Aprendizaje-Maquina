import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.utils import resample

# Cargar el dataset
df = pd.read_csv("metodosDeValidacion.csv")

# Separar características y etiquetas (suponiendo que la última columna es la etiqueta)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# I. Funciones de Scikit-Learn
# a) train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
print("\nConjunto de Entrenamiento:")
print(X_train)
print("\nConjunto de Prueba:")
print(X_test)

# b) KFold(k=6)
kf = KFold(n_splits=6, shuffle=False)
print("\nValidación Cruzada con KFold (k=6):")
for train_index, val_index in kf.split(X):
    print(f"Entrenamiento: {train_index}, Validación: {val_index}")

# c) LeaveOneOut()
loo = LeaveOneOut()
print("\nValidación Cruzada con LeaveOneOut:")
for train_index, val_index in loo.split(X):
    print(f"Entrenamiento: {train_index}, Validación: {val_index}")

# d) Bootstrap con tamaño de entrenamiento 9 (2 muestras)
print("\nValidación con Bootstrap:")
for i in range(2):
    X_resample, y_resample = resample(X, y, n_samples=9, replace=True, random_state=i)
    print(f"Muestra {i+1}:")
    print(X_resample)
