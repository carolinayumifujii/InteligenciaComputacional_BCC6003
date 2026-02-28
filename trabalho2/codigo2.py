import numpy as np

# Função para carregar os dados de um arquivo CSV
def load_data(filename):
    # Carrega o arquivo CSV, ignorando linhas de cabeçalho (se houver)
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Use skiprows=0 se não houver cabeçalho
    X = data[:, :-1]  # Todas as colunas, exceto a última (características)
    y = data[:, -1]   # Apenas a última coluna (rótulos)
    return X, y

# Carrega os dados de treinamento e teste
X_train, y_train = load_data('simpsonTreino.csv')
X_test, y_test = load_data('simpsonTeste.csv')

# Extração de características - criando novas características
mean_features_train = np.mean(X_train, axis=1, keepdims=True)
var_features_train = np.var(X_train, axis=1, keepdims=True)

mean_features_test = np.mean(X_test, axis=1, keepdims=True)
var_features_test = np.var(X_test, axis=1, keepdims=True)

# Combina as novas características com as antigas
X_train_extracted = np.hstack((X_train, mean_features_train, var_features_train))
X_test_extracted = np.hstack((X_test, mean_features_test, var_features_test))

# Salva as novas características em arquivos de texto
np.savetxt('treinamento_extraido.csv', np.hstack((X_train_extracted, y_train.reshape(-1, 1))), delimiter=',', fmt='%f')
np.savetxt('teste_extraido.csv', np.hstack((X_test_extracted, y_test.reshape(-1, 1))), delimiter=',', fmt='%f')
