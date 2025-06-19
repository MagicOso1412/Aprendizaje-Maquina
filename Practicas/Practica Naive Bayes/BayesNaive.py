import csv
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from openpyxl import Workbook
import sys
import os

def cross_validate_and_test(X, y, model_type):
    kf = KFold(n_splits=5)
    fold_accuracies = []

    for train_index, val_index in kf.split(X):
        X_train_k, X_val_k = X[train_index], X[val_index]
        y_train_k, y_val_k = y[train_index], y[val_index]

        model = GaussianNB() if model_type == 'Normal' else MultinomialNB()
        model.fit(X_train_k, y_train_k)
        y_pred_k = model.predict(X_val_k)
        acc = accuracy_score(y_val_k, y_pred_k)
        fold_accuracies.append(acc)

    return fold_accuracies

def process_dataset(X, y, dataset_name, model_types, results_table1, results_table2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    best_config = None
    best_score = -1
    best_class_metrics = None
    best_model_type = None
    best_conf_matrix = None

    for model_type in ['Normal', 'Multinomial']:
        acc_list = cross_validate_and_test(X_train, y_train, model_type)

        for i, acc in enumerate(acc_list, start=1):
            results_table1.append({
                "Dataset": dataset_name,
                "No. Pliegues": 5,
                "Distribución": model_type,
                "Pliegue": i,
                "Accuracy": acc
            })
        results_table1.append({
            "Dataset": dataset_name,
            "No. Pliegues": 5,
            "Distribución": model_type,
            "Pliegue": "Promedio",
            "Accuracy": np.mean(acc_list)
        })

        # Evaluación en prueba
        model = GaussianNB() if model_type == 'Normal' else MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results_table2.append({
            "Dataset": dataset_name,
            "Distribución": model_type,
            "Accuracy": final_accuracy
        })

        report = classification_report(y_test, y_pred, output_dict=True)
        target_class = "1" if "1" in report else list(report.keys())[0]
        class_metrics = report[target_class]

        if final_accuracy > best_score:
            best_score = final_accuracy
            best_conf_matrix = conf_matrix
            best_config = {
                "Dataset": dataset_name,
                "Distribución": model_type,
                "Accuracy": final_accuracy,
                "Clase evaluada": target_class,
                "Precision": class_metrics['precision'],
                "Recall": class_metrics['recall'],
                "F1-score": class_metrics['f1-score']
            }

    # Imprimir el mejor resultado con matriz de confusión
    print(f"=== Mejor configuración para {best_config['Dataset']} ===")
    print(f"Distribución: {best_config['Distribución']}")
    print(f"Accuracy: {best_config['Accuracy']:.4f}")
    print(f"Clase evaluada: {best_config['Clase evaluada']}")
    print(f"Precision: {best_config['Precision']:.4f}")
    print(f"Recall:    {best_config['Recall']:.4f}")
    print(f"F1-score:  {best_config['F1-score']:.4f}")
    print("Matriz de confusión:")
    print(best_conf_matrix)
    print("-" * 40)


def save_to_csv(filename, header, data):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def save_to_excel(filename, tabla1, tabla2):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Tabla1"
    ws1.append(["Dataset", "No. Pliegues", "Distribución", "Pliegue", "Accuracy"])
    for row in tabla1:
        ws1.append(row)

    ws2 = wb.create_sheet("Tabla2")
    ws2.append(["Dataset", "Distribución", "Accuracy"])
    for row in tabla2:
        ws2.append(row)

    wb.save(filename)

# Cargar CSVs manualmente sin pandas
def load_csv_iris(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
    headers = lines[0].split(',')
    data = [line.split(',') for line in lines[1:]]
    X = np.array([list(map(float, row[:-1])) for row in data])
    y = np.array([row[-1] for row in data])
    return X, y

def load_csv_emails(path):
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')
    headers = lines[0].split(',')
    data = [line.split(',') for line in lines[1:]]
    X = np.array([list(map(float, row[1:-1])) for row in data])  # Ignora ID y clase
    y = np.array([row[-1] for row in data])
    return X, y

# MAIN
tabla1 = []
tabla2 = []

X_iris, y_iris = load_csv_iris("iris.csv")
X_emails, y_emails = load_csv_emails("emails.csv")

process_dataset(X_iris, y_iris, "iris.csv", ["Normal", "Multinomial"], tabla1, tabla2)
process_dataset(X_emails, y_emails, "emails.csv", ["Normal", "Multinomial"], tabla1, tabla2)

# Guardar resultados
save_to_csv("tabla1.csv", ["Dataset", "No. Pliegues", "Distribución", "Pliegue", "Accuracy"], tabla1)
save_to_csv("tabla2.csv", ["Dataset", "Distribución", "Accuracy"], tabla2)
save_to_excel("resultados.xlsx", tabla1, tabla2)

# Mostrar las 2 mejores configuraciones (mayor accuracy en prueba)
top2 = sorted(tabla2, key=lambda x: x[2], reverse=True)[:2]

print("\n Resultados guardados en 'tabla1.csv', 'tabla2.csv' y 'resultados.xlsx'")

