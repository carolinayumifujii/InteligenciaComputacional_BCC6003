import numpy as np
import random
from collections import Counter

# Função para carregar dados de um arquivo CSV
def load_data(filename):
    # Carrega o arquivo CSV, ignorando a linha de cabeçalho, se existir
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Use skiprows=0 se não houver cabeçalho
    X = data[:, :-1]  # Todas as colunas menos a última (características)
    y = data[:, -1].astype(int)  # Última coluna como rótulos, convertida para inteiros
    return X, y

# Função para normalização Min-Max
def min_max_normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1
    return (X - X_min) / denom

# Função para normalização Z-score
def z_score_normalize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1
    return (X - X_mean) / X_std

# Função para cálculo da distância Euclidiana
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# Função para cálculo da distância Manhattan
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)

# Função k-NN
def knn_predict(X_train, y_train, X_test, k, distance_func):
    predictions = []
    for test_point in X_test:
        distances = distance_func(X_train, test_point)
        neighbor_indices = np.argsort(distances)[:k]
        neighbor_labels = y_train[neighbor_indices]
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)

# Função para cálculo da acurácia
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

# Função para dividir aleatoriamente o conjunto de treinamento
def split_training_set(X, y, fraction):
    indices = list(range(len(X)))
    random.shuffle(indices)
    split_point = int(len(X) * fraction)
    train_indices = indices[:split_point]
    return X[train_indices], y[train_indices]

# Função para avaliar o k-NN
def evaluate_knn(X_train, y_train, X_test, y_test, k_values, distance_funcs, normalize_func, split_fractions):
    X_train_norm = normalize_func(X_train)
    X_test_norm = normalize_func(X_test)
    
    for fraction in split_fractions:
        X_train_frac, y_train_frac = split_training_set(X_train_norm, y_train, fraction)
        print(f"\n\ndivisão do conjunto de treinamento: {int(fraction * 100)}%")
        
        for dist_name, dist_func in distance_funcs:
            print(f"\n{dist_name}")
            for k in k_values:
                y_pred = knn_predict(X_train_frac, y_train_frac, X_test_norm, k, dist_func)
                acc = calculate_accuracy(y_test, y_pred)
                print(f"k={k}, acuracia: {acc:.2f}%")

def main():
    # Carrega os dados de treinamento e teste
    X_train, y_train = load_data('simpsonTreino.csv')
    X_test, y_test = load_data('simpsonTeste.csv')
    
    # Parâmetros para k-NN
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    distance_funcs = [("distância Euclidiana", euclidean_distance),
                      ("distância Manhattan", manhattan_distance)]
    normalize_funcs = [("Min-Max", min_max_normalize),
                       ("Z-score", z_score_normalize)]
    split_fractions = [0.25, 0.50, 1.00]
    
    # Avaliação usando características originais
    print("** parte 1: usando as features disponíveis **")
    for norm_name, norm_func in normalize_funcs:
        print(f"\n\nnormalização: {norm_name}")
        evaluate_knn(X_train, y_train, X_test, y_test, k_values, distance_funcs, norm_func, split_fractions)
    
    # Carrega dados com features extraídas
    X_train_extracted, y_train_extracted = load_data('treinamento_extraido.csv')
    X_test_extracted, y_test_extracted = load_data('teste_extraido.csv')
    
    print("\n\n** parte 2: usando as features extraídas **")
    for norm_name, norm_func in normalize_funcs:
        print(f"\n\nnormalização: {norm_name}")
        evaluate_knn(X_train_extracted, y_train_extracted, X_test_extracted, y_test_extracted, k_values, distance_funcs, norm_func, split_fractions)

if __name__ == "__main__":
    main()
