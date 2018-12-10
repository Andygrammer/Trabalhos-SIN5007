from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from training_test import TrainnigTestFolds
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# inicializa totais das estimativas
tot_precision = 0
tot_recall = 0
tot_accuracy = 0

# kpartes
kPartes = 17

# informa o dataset a ser utilizado via construtor
ttf = TrainnigTestFolds("dataset_dummies.csv")
# ttf = TrainnigTestFolds("dsComMediasBinariosSemDummies.csv")



# escolha do kernel
# svclassifier = SVC(kernel='linear')#linear(apenas para comparacao)
# svclassifier = SVC(kernel='poly', degree=7)#Polynomial
svclassifier = SVC(kernel='rbf', C=1000)  # Gaussian
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

    print(X_test.shape)
    # normaliza
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(X_test.shape)

    # PCA
    pca = PCA(n_components=12).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # treina
    # svclassifier = SVC(kernel='rbf', C=1000)  # Gaussian
    svclassifier.fit(X_train, y_train)

    # estimacao dos testes
    y_pred = svclassifier.predict(X_test)

    # pega a precisao e recall da classe prevista como noRecorrenceEvents
    # se quiser mudar para recorrenceEventes, trocar os indices dos registros de 0 para 1


    tot_precision += precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1.0)[0]
    tot_recall += precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1.0)[1]
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