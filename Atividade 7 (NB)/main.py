import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from training_test import TrainnigTestFolds
from sklearn.metrics import recall_score, precision_score, accuracy_score
import os

def getPCA(fileToPCA, fileFromPCA, limiar):

    #Delete the Old File
    if os.path.exists(fileFromPCA):
        os.remove(fileFromPCA)
    #Open
    toBeProcess = pd.read_csv(fileToPCA)
    classes = toBeProcess['class']

    #Normalize
    scaler = StandardScaler().fit(toBeProcess)
    std = scaler.transform(toBeProcess)

    #PCA
    pca = PCA(n_components=12).fit(std)

    index = 1
    total = 0
    for component in pca.explained_variance_ratio_:
        if total > limiar:
            break
        index += 1
        total += component

    pca = PCA(n_components=index).fit(std)
    processed = pca.transform(std)

    #Save to csv
    data = pd.DataFrame(processed)
    csv = pd.concat([data, classes], axis=1)
    csv.to_csv(fileFromPCA)

# Create model
def main(ffunction, fileName, kPartes, ttf):
    # inicializa totais das estimativas
    tot_precision = 0
    tot_recall = 0
    tot_accuracy = 0

    for i in range(0, kPartes - 1):

        df_training = ttf.k_df_training(i)
        df_test = ttf.k_df_test(i)
        target_training = df_training['class']
        target_test = df_test['class']

        # separa os atributos e as classes de teste e treinamento
        x_train = df_training.drop('class', axis=1)
        y_train = target_training
        x_test = df_test.drop('class', axis=1)
        y_test = target_test


        model = ffunction().fit(x_train, y_train)

        predicted = model.predict(x_test)

        recall = recall_score(y_test, predicted)
        precision = precision_score(y_test, predicted)
        accuracy = accuracy_score(y_test, predicted)

        tot_precision += (precision/kPartes)
        tot_recall += (recall/kPartes)
        tot_accuracy += (accuracy/kPartes)



    print("Testing Accuracy: {:.4f} Recall: {:.4f} Precision: {:.4f}\n\n\n".format(tot_accuracy,
                                                                                    tot_recall,
                                                                                    tot_precision));
    file = open(fileName, "+a")
    file.write("Testing Accuracy: {:.4f} Recall: {:.4f} Precision: {:.4f}\n\n\n".format(tot_accuracy,
                                                                                        tot_recall,
                                                                                        tot_precision))
    file.close()


if __name__ == "__main__":
    # kpartes
    kPartes = 17

    # File
    fileFromDummies = "Data/dataset_dummies.csv"
    fileFromPCA = "Data/dataset_dummies_pca.csv"
    fileFromSelection = "Data/dataset_selection.csv"

    #Generate PCA
    getPCA(fileFromDummies, fileFromPCA, 0.75)

    fileToSaveResult = "Result/ResultProcessing.txt"

    #Para testar, apenas descomente um ()
    #With PCA
    #ttf = TrainnigTestFolds(fileFromPCA)

    #With Dummies
    ttf = TrainnigTestFolds(fileFromDummies)

    #With Selection
    #ttf = TrainnigTestFolds(fileFromSelection)

    ffunction = GaussianNB
    #ffunction = MultinomialNB  #Nao Funciona com PCA pelos valores negativos
    main(ffunction, fileToSaveResult, kPartes, ttf)

    #Extra Info
    #https://www.quora.com/What-is-the-difference-between-the-the-Gaussian-Bernoulli-Multinomial-and-the-regular-Naive-Bayes-algorithms
    #