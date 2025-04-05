import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def bgdMulti(archivo, iteraciones, pesos, alpha):
    df = pd.read_csv(archivo)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=0)

    # ------------Gradiente Descendente-------------
    weights = []
    w = pesos
    print('w\n')
    # print('-------------w-------------')
    print(w)
    for j in range(iteraciones):
        for i in range(5):
            x = X_train.iloc[:, i]
            suma = np.sum((w[i] * x - Y_train) * x)
            w[i] = w[i] - 2 * alpha * suma
        weight = np.array(w)
        weights.append(weight)
        print(f'Iteración {j + 1}: {weight}')
    # ------------Gradiente Descendente-------------

    # -----------Pruebas con los pesos obtenidos------------
    print('y_test\n')
    # print('\n---------y_test-----------')
    print(Y_test)

    # print('\n---------y_predict--------')
    print('y_predict\n')
    predicciones = []
    for i, vectorPesos in enumerate(weights):
        predict = np.array([])
        for j in range(3):
            y_predict = np.sum(X_test.iloc[j, :] * vectorPesos)
            predict = np.append(predict, y_predict)
        predicciones.append(predict)
        print(f'Iteración {i + 1}: {predict}')
    # -----------Pruebas con los pesos obtenidos------------

    # ----------------Cálculo del error estimado-----------
    # print('\n' + '--------------Error de estimación---------------')
    print('Error de estimación\n')
    errores = []
    for i, prediccion in enumerate(predicciones):
        error = np.sum(np.abs(prediccion - Y_test))
        errores.append(error)
        print(f'Iteración {i + 1}: {error}')
    # ----------------Cálculo del error estimado-----------
    graficaErrores(errores)


def graficaErrores(datos):
    colors = ['gold', 'yellowgreen', 'deeppink', 'mediumpurple']
    # Cambia el color de cada punto de error utilizando colors
    for i in range(len(datos)):
        plt.scatter(i, datos[i], color=colors[i % len(colors)], label='Error estimación' if i == 0 else "")

    plt.ylabel('Error estimación')
    plt.xlabel('Iteración')
    plt.title('Error de estimación por iteración')
    plt.legend
    plt.show()


def graficaPredicciones(X_test, Y_test, predicciones):
    plt.figure(figsize=(12, 5))
    colors = ['gold', 'yellowgreen', 'deeppink', 'mediumpurple']
    # plt.scatter(X_test,Y_test, label = 'Datos de prueba')

    # Cambia el color de cada punto de prueba utilizando colors
    for i in range(len(X_test)):
        plt.scatter(X_test.iloc[i], Y_test.iloc[i], color='lightseagreen', label='Datos de prueba' if i == 0 else "")
    # Dibuja las líneas de predicción con colores alternados para cada iteración
    for i, pred in enumerate(predicciones):
        plt.plot(X_test, pred, linestyle='--', color=colors[i % len(colors)], label=f'Iteración {i + 1}')

    plt.title('Regresion lineal')
    plt.xlabel('Terreno (m2)')
    plt.ylabel('Precio (MDP)')
    # plt.legend()
    plt.show()


# descGradient('casas.csv', 4, 0, 0.00000007) #Funciona bien con 5 iteraciones y paso 0.05
bgdMulti('Dataset_multivariable.csv', 4, [0, 0, 0, 0, 0], 0.000006)