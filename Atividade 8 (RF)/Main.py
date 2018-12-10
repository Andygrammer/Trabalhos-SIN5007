from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from training_test import TrainnigTestFolds
import pandas as pd
from sklearn.preprocessing import StandardScaler
import h2o

# inicializa totais das estimativas
tot_precision = 0
tot_recall = 0
tot_accuracy = 0

# kpartes
kPartes = 17

# informa o dataset a ser utilizado via construtor
ttf = TrainnigTestFolds("breast_cancer_original.csv")
# ttf = TrainnigTestFolds("dsComMediasBinariosSemDummies.csv")


# escolha do kernel
# svclassifier = SVC(kernel='linear')#linear(apenas para comparacao)
# svclassifier = SVC(kernel='poly', degree=7)#Polynomial
rf_classifier = RandomForestClassifier(n_estimators=500)
# svclassifier = SVC(kernel='sigmoid')#Sigmoid


# para cada conjunto de treinamento e teste dados pelo k-fold
for i in range(0, kPartes - 1):
    print(i)
    df_training = ttf.k_df_training(i)
    df_test = ttf.k_df_test(i)

    # separa os atributos e as classes de teste e treinamento
    X_train = df_training.drop('class', axis=1)
    y_train = df_training['class']
    X_test = df_test.drop('class', axis=1)
    y_test = df_test['class']

    rf_classifier.fit(X_train, y_train)

    # estimacao dos testes
    y_pred = rf_classifier.predict(X_test)

    # pega a precisao e recall da classe prevista como noRecorrenceEvents
    # se quiser mudar para recorrenceEventes, trocar os indices dos registros de 0 para 1


    tot_precision += precision_recall_fscore_support(y_test, y_pred, average='binary')[0]
    tot_recall += precision_recall_fscore_support(y_test, y_pred, average='binary')[1]
    tot_accuracy += accuracy_score(y_test, y_pred)

    # totNoRecPrecision += precision_recall_fscore_support(y_test, y_pred)
    # totNoRecRecall += precision_recall_fscore_support(y_test, y_pred)
    # totAccuracy += accuracy_score(y_test, y_pred)

# gera a media entre as k-partes
mediaNoRecPrecision = tot_precision / kPartes
mediaNoRecRecall = tot_recall / kPartes
mediaNoRecAccurary = tot_accuracy / kPartes

# imprime as medidas
print("Precisão  : " + str(mediaNoRecPrecision))
print("Recall    : " + str(mediaNoRecRecall))
print("Acurácia  : " + str(mediaNoRecAccurary))