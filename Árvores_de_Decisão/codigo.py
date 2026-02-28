import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Carregar o dataset Titanic localmente
# Certifique-se de que o caminho está correto
titanic = pd.read_csv('/Users/carolina/Desktop/IC/Árvores_de_Decisão/titanic.csv')

# Pré-processamento dos dados
titanic = titanic.drop(columns=['Name', 'Ticket', 'Cabin'])
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic['Embarked'] = titanic['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
titanic = titanic.fillna(titanic.mean())

X = titanic.drop(columns='Survived')
y = titanic['Survived']

# Dividir os dados em conjunto de treinamento (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de árvore de decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Visualizar a árvore de decisão
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()

# Avaliar o modelo antes da poda
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"Treinamento - Acurácia: {accuracy_train}, F1-score: {f1_train}")
print(f"Teste - Acurácia: {accuracy_test}, F1-score: {f1_test}")

# Poda da árvore de decisão
model_podada = DecisionTreeClassifier(max_depth=3, random_state=42)
model_podada.fit(X_train, y_train)

# Visualizar a árvore de decisão podada
plt.figure(figsize=(20,10))
plot_tree(model_podada, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()

# Avaliar o modelo após a poda
y_pred_train_podada = model_podada.predict(X_train)
y_pred_test_podada = model_podada.predict(X_test)
accuracy_train_podada = accuracy_score(y_train, y_pred_train_podada)
accuracy_test_podada = accuracy_score(y_test, y_pred_test_podada)
f1_train_podada = f1_score(y_train, y_pred_train_podada)
f1_test_podada = f1_score(y_test, y_pred_test_podada)

print(f"Treinamento após poda - Acurácia: {accuracy_train_podada}, F1-score: {f1_train_podada}")
print(f"Teste após poda - Acurácia: {accuracy_test_podada}, F1-score: {f1_test_podada}")
