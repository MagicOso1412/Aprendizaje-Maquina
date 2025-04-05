import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Datos originales
X_data = np.array([440.0, 616.0, 381.0, 963.0, 431.0, 255.0, 594.0, 625.0, 708.0, 468.0])
y_data = np.array([1.01, 1.42, 0.88, 2.21, 0.99, 0.59, 1.37, 1.44, 1.63, 1.08])

# División de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)

# Función de predicción
def predecir(X, w):
    return X * w

# Error absoluto medio
def calcular_error(y_pred, y_real):
    return np.mean(np.abs(y_pred - y_real))

# Gradiente descendente corregido
def gradiente_descendente(X_train, y_train, X_test, y_test, w_inicial, alpha, iteraciones):
    w = w_inicial
    errores = []
    predicciones = []
    pesos = []

    for i in range(iteraciones):
        y_pred_train = predecir(X_train, w)
        gradiente = -2 * np.sum((y_train - y_pred_train) * X_train)
        w -= alpha * gradiente

        y_pred_test = predecir(X_test, w)
        err = calcular_error(y_pred_test, y_test)

        errores.append(err)
        predicciones.append(y_pred_test)
        pesos.append(w)

        print(f"Iteración {i+1}: peso = {w:.4f}, error = {err:.4f}")

    return pesos, predicciones, errores

# Gráficas
def graficar_resultados(X_test, y_test, predicciones, errores):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_test, y_test, color='black', label="Datos reales")
    for i, y_pred in enumerate(predicciones):
        plt.plot(X_test, y_pred, label=f"Iter {i+1}")
    plt.title("Gráfica 1: Predicción en cada iteración")
    plt.xlabel("Terreno (m2)")
    plt.ylabel("Precio (MDP)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(errores)+1), errores, marker='o')
    plt.title("Gráfica 2: Error por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Error absoluto medio")
    plt.grid(True)
    plt.show()

# Parámetros
iteraciones = 4
alpha = 0.00000007
peso_inicial = 0

# Ejecutar
pesos, predicciones, errores = gradiente_descendente(X_train, y_train, X_test, y_test, peso_inicial, alpha, iteraciones)

# Resultados finales
print("\nPesos por iteración:")
for i, w in enumerate(pesos, 1):
    print(f"Iteración {i}: peso = {w:.4f}")

print(f"\nPenúltimo peso: {pesos[-2]:.4f}")
print(f"Peso final: {pesos[-1]:.4f}")

graficar_resultados(X_test, y_test, predicciones, errores)
