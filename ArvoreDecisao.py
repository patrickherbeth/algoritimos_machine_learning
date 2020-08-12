import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pydot
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from ipywidgets import interactive
from IPython.display import SVG, display
from graphviz import Source
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Carregando a base de dados:
df_diabetes = pd.read_csv('sample_data/diabetes.csv')
print(df_diabetes.head())

# Informaçoes sobre a base:
df_diabetes.info()

# Dividindo os dados em treino e teste:
X_train, X_test, y_train, y_test = train_test_split(df_diabetes.drop('Outcome', axis=1), df_diabetes['Outcome'],
                                                    test_size=0.3)

# Verificando as formas dos dados:
X_train.shape, X_test.shape
((537, 8), (231, 8))

y_train.shape, y_test.shape
((537,), (231,))

# Instânciando o objeto classificador:
clf = DecisionTreeClassifier()

# Treinando o modelo de arvore de decisão:
clf = clf.fit(X_train, y_train)

# Verificando as features mais importantes para o modelo treinado:
clf.feature_importances_

# O código acima nos retorna um array com o valor de cada variável:

np.array([0.06285337, 0.36516388, 0.10650456, 0.0510792, 0.04390734,
          0.18012907, 0.10751651, 0.08284606])

for feature, importancia in zip(df_diabetes.columns, clf.feature_importances_):
    print("{}:{}".format(feature, importancia))

resultado = clf.predict(X_test)
print(resultado)

# Resultado do classification_report:

print(metrics.classification_report(y_test, resultado))

# Renderizando a árvore de forma gráfica:

dot_data = export_graphviz(
    clf,
    out_file="plot.dot",
    feature_names=df_diabetes.drop('Outcome', axis=1).columns,
    class_names=['0', '1'],
    filled=True, rounded=True,
    proportion=True,
    node_ids=True,
    rotate=False,
    label='all',
    special_characters=True
)

from IPython.display import SVG

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph

# Renderizando a árvore de forma interativa:

# feature matrix
X, y = df_diabetes.drop('Outcome', axis=1), df_diabetes['Outcome']

# feature labels
features_label = df_diabetes.drop('Outcome', axis=1).columns

# class label
class_label = ['0', '1']

def plot_tree(crit, split, depth, min_samples_split, min_samples_leaf=0.2):
    estimator = DecisionTreeClassifier(
        random_state=0
        , criterion=crit
        , splitter=split
        , max_depth=depth
        , min_samples_split=min_samples_split
        , min_samples_leaf=min_samples_leaf
    )
    estimator.fit(X, y)
    graph = Source(export_graphviz(estimator
                                   , out_file=None
                                   , feature_names=features_label
                                   , class_names=class_label
                                   , impurity=True
                                   , filled=True))
    display(SVG(graph.pipe(format='svg')))
    return estimator

inter = interactive(plot_tree
                    , crit=["gini", "entropy"]
                    , split=["best", "random"]
                    , depth=[1, 2, 3, 4, 5, 10, 20, 30]
                    , min_samples_split=(1, 5)
                    , min_samples_leaf=(1, 5))

display(inter)

# Visualizando as fronteiras criadas pela arvore:

def visualize_fronteiras(msamples_split, max_depth):
    X = df_diabetes[['Glucose', 'Insulin']].values
    y = df_diabetes.Outcome.values
    clf = DecisionTreeClassifier(min_samples_split=msamples_split, max_depth=max_depth)
    tree = clf.fit(X, y)

    plt.figure(figsize=(16, 9))
    plot_decision_regions(X, y, clf=tree, legend=2)

    plt.xlabel('Glucose')
    plt.ylabel('Insulin')
    plt.title('Decision Tree')
    plt.show()


# Chamando a função criada anteriormente:

visualize_fronteiras(2, max_depth=30)
