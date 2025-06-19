import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# 1. Cargar el dataset Iris
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
class_names = iris.target_names

# 2. Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrenamiento One vs All
svms = []
support_vectors = {}
N = {}

for i, class_name in enumerate(class_names):
    y_binary = np.where(y_train == i, 1, 0)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_binary)
    svms.append(clf)

    # Calcular N, c (promedio de clase), c¬ (promedio de no clase), c (promedio general)
    class_mask = y_binary == 1
    not_class_mask = y_binary == 0
    N[class_name] = sum(class_mask)
    c = X_train[class_mask].mean().values
    c_not = X_train[not_class_mask].mean().values
    c_total = X_train.mean().values
    support_vectors[class_name] = (N[class_name], c, c_not, c_total)


# === Función de clasificación para cualquier conjunto ===
def clasificar(X_data):
    probabilities = []
    for x in X_data.values:
        class_probs = []
        for i, clf in enumerate(svms):
            proba = clf.predict_proba([x])[0][1]
            class_probs.append(proba)
        probabilities.append(class_probs)
    predictions = []
    for prob in probabilities:
        if prob[0] == prob[1] or prob[1] == prob[2] or prob[0] == prob[2]:
            predictions.append(-1)  # indefinida
        else:
            predictions.append(np.argmax(prob))
    return predictions


# 4. Clasificación del conjunto de entrenamiento
train_predictions = clasificar(X_train)
train_valid_indices = [i for i, p in enumerate(train_predictions) if p != -1]
y_train_valid = y_train.values[train_valid_indices]
y_train_pred_valid = [train_predictions[i] for i in train_valid_indices]

print("\n=== Reporte de Clasificación (ENTRENAMIENTO) ===")
print(classification_report(y_train_valid, y_train_pred_valid, target_names=class_names))
print("=== Matriz de Confusión (ENTRENAMIENTO) ===")
print(confusion_matrix(y_train_valid, y_train_pred_valid))

# 5. Clasificación del conjunto de prueba
test_predictions = clasificar(X_test)
test_valid_indices = [i for i, p in enumerate(test_predictions) if p != -1]
y_test_valid = y_test.values[test_valid_indices]
y_test_pred_valid = [test_predictions[i] for i in test_valid_indices]

print("\n=== Reporte de Clasificación (PRUEBA) ===")
print(classification_report(y_test_valid, y_test_pred_valid, target_names=class_names))
print("=== Matriz de Confusión (PRUEBA) ===")
print(confusion_matrix(y_test_valid, y_test_pred_valid))

# 6. Mostrar detalles de entrenamiento
for class_name in class_names:
    N_val, c, c_not, c_total = support_vectors[class_name]
    print(f"\nClase: {class_name}")
    print(f"N: {N_val}")
    print(f"c (promedio clase): {c}")
    print(f"c¬ (promedio no clase): {c_not}")
    print(f"c total: {c_total}")

# === Clasificación detallada de los datos de entrenamiento ===
print("\n=== Clasificación de los datos de ENTRENAMIENTO ===")
for idx, pred in enumerate(train_predictions):
    clase = class_names[pred] if pred != -1 else "Indefinida"
    print(f"Dato {idx} → {clase}")
