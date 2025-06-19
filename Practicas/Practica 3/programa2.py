import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar dataset
data = pd.read_csv('cal_housing.csv')

# Separar características y target
X = data.drop(columns=["medianHouseValue"])
y = data["medianHouseValue"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=0
)

# Función para ejecutar el flujo: elevar, escalar (si aplica), y ajustar modelo
def ejecutar_modelo(nombre, X_train, X_test, y_train, y_test,
                    grado=1, escalador=None):
    # Elevar
    poly = PolynomialFeatures(degree=grado)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    # Escalar si se especifica
    if escalador == 'standard':
        scaler = StandardScaler()
        X_train_poly = scaler.fit_transform(X_train_poly)
        X_test_poly = scaler.fit_transform(X_test_poly)
        nombre += " + StandardScaler"
    elif escalador == 'robust':
        scaler = RobustScaler()
        X_train_poly = scaler.fit_transform(X_train_poly)
        X_test_poly = scaler.fit_transform(X_test_poly)
        nombre += " + RobustScaler"

    # Regresor
    modelo = LinearRegression()
    modelo.fit(X_train_poly, y_train)
    y_pred = modelo.predict(X_test_poly)

    # Resultados
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"Modelo": nombre, "MSE": mse, "R²": r2}

# Ejecutar todos los modelos
resultados = []

# Lineal (grado 1, sin escalar)
resultados.append(ejecutar_modelo("Lineal", X_train, X_test, y_train, y_test, grado=1))

# Polinomial grado 2
resultados.append(ejecutar_modelo("Polinomial grado 2", X_train, X_test, y_train, y_test, grado=2))
resultados.append(ejecutar_modelo("Polinomial grado 2", X_train, X_test, y_train, y_test, grado=2, escalador='standard'))
resultados.append(ejecutar_modelo("Polinomial grado 2", X_train, X_test, y_train, y_test, grado=2, escalador='robust'))

# Polinomial grado 3
resultados.append(ejecutar_modelo("Polinomial grado 3", X_train, X_test, y_train, y_test, grado=3))
resultados.append(ejecutar_modelo("Polinomial grado 3", X_train, X_test, y_train, y_test, grado=3, escalador='standard'))
resultados.append(ejecutar_modelo("Polinomial grado 3", X_train, X_test, y_train, y_test, grado=3, escalador='robust'))

# Mostrar resultados
df_resultados = pd.DataFrame(resultados)
print(df_resultados)
