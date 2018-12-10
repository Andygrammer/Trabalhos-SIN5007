import kfold as folds
import pandas as pd

"""modulo que utilizando os k datasets retornados pelo k-fold, separa o de teste e uno os de treinamento
"  ele retorna apenar um teste e um treinamento(unindo todos os registros exceto teste), portanto
"  deve ser chamado k vezes pelo utilizador"""

class TrainnigTestFolds:
    def __init__(self,df):   
        #divide o dataset em k partes utilizando k-fold
        df = pd.read_csv(df)
        class_column = 'class'
        k = 17
        self.folds_list = folds.k_fold(df, k, class_column)
    
    
    #cria um dataframe de treinamento com todas as partes, exceto a de treinamento da rodada
    def k_df_training(self,test):
        #deleta o dataframe de teste
        training_list = self.folds_list
        #training_list.__delitem__(test) 
        
        #inicia o dataframe de treinamento com o primeiro(correto) do dataset
        #fiz isso pq é necessario inicia-lo e nao consegui achar uma forma melhor
        if test == 0:
            df_training = training_list[1]
            i =2
        else:
            df_training = training_list[0]
            i = 1
        
      
        #adiciona registros de todas as partes, exceto a de teste
        for i in range(i, len(self.folds_list)):        
           if i != test :
               df_training = df_training.append(training_list[i])
           
        return df_training
    
    #cria o dataframe de teste
    def k_df_test(self,test):
        df_test = self.folds_list.__getitem__(test)
        return df_test
        
        
    
