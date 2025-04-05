import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.utils import resample
import numpy as np

# Cargar el dataset
df = pd.read_csv("metodosDeValidacion.csv")

# Separar características y etiquetas (suponiendo que la última columna es la etiqueta)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# I. Funciones de Scikit-Learn
# a) train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
print("\nConjunto de Entrenamiento X:")
print(X_train.transpose())
print("\nConjunto de Prueba X:")
print(X_test.transpose())
print("\nConjunto de Entrenamiento Y:")
print(y_train.transpose())
print("\nConjunto de Prueba Y:")
print(y_test.transpose())

# b) KFold(k=6)
kf = KFold(n_splits=6, shuffle=False)
print("\nValidación Cruzada con KFold (k=6):")
for train_index, val_index in kf.split(X_train):
    print("Entrenamiento X:")
    print(X_train[train_index].transpose())
    print("Validación X:")
    print(X_train[val_index].transpose())
    print("Entrenamiento Y:")
    print(y_train[train_index].transpose())
    print("Validación Y:")
    print(y_train[val_index].transpose())

# c) LeaveOneOut()
loo = LeaveOneOut()
print("\nValidación Cruzada con LeaveOneOut:")
for train_index, val_index in loo.split(X_train):
    print("Entrenamiento X:")
    print(X_train[train_index].transpose())
    print("Validación X:")
    print(X_train[val_index].transpose())
    print("Entrenamiento Y:")
    print(y_train[train_index].transpose())
    print("Validación Y:")
    print(y_train[val_index].transpose())

# d) Bootstrap con tamaño de entrenamiento 9 (2 muestras)
print("\nValidación con Bootstrap:")
print("Entrenamiento con Bootstrap:")
print(X_train.transpose())
print(y_train.transpose())
for i in range(2):
    X_resample, y_resample = resample(X_train, y_train, n_samples=9, replace=True, random_state=i)

    # Convertir a tuplas para poder comparar correctamente
    X_resample_set = set(map(tuple, X_resample))
    X_out_of_sample = np.array([x for x in X_train if tuple(x) not in X_resample_set])
    y_out_of_sample = np.array([y_train[i] for i in range(len(y_train)) if tuple(X_train[i]) not in X_resample_set])

    print(f"Muestra {i + 1} (Entrenamiento X):")
    print(X_resample.transpose())
    print(f"Muestra {i + 1} (Entrenamiento Y):")
    print(y_resample.transpose())
    print(f"Conjunto de Prueba {i + 1} (X no seleccionado en Bootstrap):")
    print(X_out_of_sample.transpose())
    print(f"Conjunto de Prueba {i + 1} (Y no seleccionado en Bootstrap):")
    print(y_out_of_sample.transpose())

