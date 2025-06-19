import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import operator

# === ENTRADA ===
archivo = "datos.csv"
n_iteraciones = int(input("Número de iteraciones para SGD: "))
alpha = float(input("Valor de alpha (tasa de aprendizaje) para SGD: "))

# === CARGA Y DIVISIÓN DEL DATASET ===
datos = pd.read_csv(archivo)
X = datos[["x"]].values
y = datos["y"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

# === FUNCIONES DE REGRESIÓN Y GRÁFICA ===
def entrenar_y_evaluar(modelo, nombre, X_train, y_train, X_test, y_test, poly=None):
    if poly:
        X_train_t = poly.fit_transform(X_train)
        X_test_t = poly.transform(X_test)
    else:
        X_train_t = X_train
        X_test_t = X_test

    modelo.fit(X_train_t, y_train)
    y_pred = modelo.predict(X_test_t)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Ordenar para graficar la curva de predicción (solo para polinomiales)
    if poly:
        sort_idx = np.argsort(X_test[:, 0])
        X_sorted = X_test[sort_idx]
        y_sorted = y_pred[sort_idx]
        y_real_sorted = y_test[sort_idx]
    else:
        X_sorted = X_test
        y_sorted = y_pred
        y_real_sorted = y_test

    plt.figure()
    plt.scatter(X_test, y_test, color="blue", label="Datos de prueba")
    plt.plot(X_sorted, y_sorted, color="red", label="Predicción")
    plt.title(nombre)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"--- {nombre} ---")
    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}\n")


# === REGRESIONES ===

# Lineal con OLS
modelo_ols = LinearRegression()
entrenar_y_evaluar(modelo_ols, "Regresión Lineal OLS", X_train, y_train, X_test, y_test)

# Polinomial grado 2 con OLS
poly2 = PolynomialFeatures(degree=2)
modelo_poly2_ols = LinearRegression()
entrenar_y_evaluar(modelo_poly2_ols, "Regresión Polinomial Grado 2 OLS", X_train, y_train, X_test, y_test, poly=poly2)

# Polinomial grado 3 con OLS
poly3 = PolynomialFeatures(degree=3)
modelo_poly3_ols = LinearRegression()
entrenar_y_evaluar(modelo_poly3_ols, "Regresión Polinomial Grado 3 OLS", X_train, y_train, X_test, y_test, poly=poly3)

# Lineal con SGD
modelo_sgd = SGDRegressor(max_iter=n_iteraciones,learning_rate="constant", eta0=alpha)
entrenar_y_evaluar(modelo_sgd, "Regresión Lineal SGD", X_train, y_train, X_test, y_test)

# Polinomial grado 2 con SGD
modelo_poly2_sgd = SGDRegressor(max_iter=n_iteraciones, learning_rate="constant", eta0=alpha)
entrenar_y_evaluar(modelo_poly2_sgd, "Regresión Polinomial Grado 2 SGD", X_train, y_train, X_test, y_test, poly=poly2)

# Polinomial grado 3 con SGD
modelo_poly3_sgd = SGDRegressor(max_iter=n_iteraciones, learning_rate="constant", eta0=alpha)
entrenar_y_evaluar(modelo_poly3_sgd, "Regresión Polinomial Grado 3 SGD", X_train, y_train, X_test, y_test, poly=poly3)
