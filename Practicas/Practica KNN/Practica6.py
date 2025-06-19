import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -------------------- Cargar Datos -------------------- #
def cargar_datasets():
    iris = pd.read_csv('iris.csv')
    emails = pd.read_csv('emails.csv')

    X_iris = iris.iloc[:, :-1]
    y_iris = iris.iloc[:, -1]

    X_emails = emails.iloc[:, 1:-1]  # Ignora columna ID
    y_emails = emails.iloc[:, -1]

    return (X_iris, y_iris), (X_emails, y_emails)

# -------------------- División de Datos -------------------- #
def dividir(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

# -------------------- Validación Cruzada Manual -------------------- #
def validacion_cruzada_manual(X, y, vecinos, pesos):
    kf = KFold(n_splits=3)
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        modelo = KNeighborsClassifier(n_neighbors=vecinos, weights=pesos)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    return accuracies, np.mean(accuracies)

def mostrar_tabla_validacion(nombre, X, y):
    configuraciones = [(1, 'uniform'), (10, 'uniform'), (10, 'distance')]
    print(f"\n--- Tabla 1: Validación cruzada - {nombre} ---")
    for vecinos, pesos in configuraciones:
        accs, prom = validacion_cruzada_manual(X, y, vecinos, pesos)
        for i, acc in enumerate(accs):
            print(f"{vecinos} vecinos - {pesos} - Pliegue {i+1}: Accuracy = {acc:.4f}")
        print(f"Promedio: {prom:.4f}\n")

# -------------------- Prueba Final K-NN -------------------- #
def prueba_final_knn(X_train, X_test, y_train, y_test, vecinos, pesos, nombre):
    modelo = KNeighborsClassifier(n_neighbors=vecinos, weights=pesos)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    print(f"\n--- Tabla 2: Reporte K-NN - {nombre} ---")
    print(classification_report(y_test, predicciones))
    ConfusionMatrixDisplay.from_predictions(y_test, predicciones).plot()
    plt.title(f'Matriz de Confusión - K-NN - {nombre}')
    plt.show()

# -------------------- Prueba Final Naïve Bayes -------------------- #
def prueba_final_bayes(X_train, X_test, y_train, y_test, nombre):
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    print(f"\n--- Tabla 2: Reporte Naïve Bayes - {nombre} ---")
    print(classification_report(y_test, predicciones))
    ConfusionMatrixDisplay.from_predictions(y_test, predicciones).plot()
    plt.title(f'Matriz de Confusión - Naïve Bayes - {nombre}')
    plt.show()

# -------------------- Main -------------------- #
def main():
    (X_iris, y_iris), (X_emails, y_emails) = cargar_datasets()

    # Tabla 1: Validación cruzada
    mostrar_tabla_validacion("iris.csv", X_iris, y_iris)
    mostrar_tabla_validacion("emails.csv", X_emails, y_emails)

    # Dividir conjuntos
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = dividir(X_iris, y_iris)
    X_train_emails, X_test_emails, y_train_emails, y_test_emails = dividir(X_emails, y_emails)

    # Tabla 2: Pruebas finales Naïve Bayes
    prueba_final_bayes(X_train_iris, X_test_iris, y_train_iris, y_test_iris, "iris.csv")
    prueba_final_bayes(X_train_emails, X_test_emails, y_train_emails, y_test_emails, "emails.csv")

    # Tabla 2: Pruebas finales K-NN
    prueba_final_knn(X_train_iris, X_test_iris, y_train_iris, y_test_iris, vecinos=10, pesos='distance', nombre="iris.csv")
    prueba_final_knn(X_train_emails, X_test_emails, y_train_emails, y_test_emails, vecinos=10, pesos='uniform', nombre="emails.csv")

if __name__ == "__main__":
    main()
