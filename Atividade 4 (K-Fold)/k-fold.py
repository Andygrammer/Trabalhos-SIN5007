import pandas as pd
from sklearn.utils import shuffle


def k_fold(df: pd.DataFrame, k, class_column, shuffle_df=True, stratify_df=True):
    """
    Chamada para o k-fold.
    :param df: dataset de interesse.
    :param k: numero de grupos em que o dataset sera dividido
    :param class_column: nome da classe por onde o dataset sera agrupado, se estratificado
    :param shuffle_df: default: True. Se verdadeiro, misturar as instancias antes do processamento.
    :param stratify_df: default: True. Se verdadeiro, ordenar o dataset pela classe, a fim de deixar a divisao com valores de classe balanceados.
    :return: lista de tamanho k com varios datasets, cada um sendo um fold.
    """
    class_values_list = list(df[class_column].unique())
    if shuffle_df:
        df = shuffle(df)
    if stratify_df:
        df = __stratify_dataframe(df, class_column, class_values_list)
    folds_list = __k_fold(df, k)
    return folds_list

def __create_folds_list(k, df_colums):
    """Cria os folds, datasets vazios com mesma estrutura de coluna."""
    folds_list = []
    for i in range(0, k):
        k_fold = pd.DataFrame(data=None, columns=df_colums)
        folds_list.append(k_fold)
    return folds_list

def __k_fold(df: pd.DataFrame, k):
    """Apos criar os folds, preenche com o conteudo do dataset original."""
    folds_list = __create_folds_list(k, df.columns)
    for i in range(0, len(df)):
        folds_list[i % k] = folds_list[i % k].append(df.iloc[i])
    return folds_list

def __stratify_dataframe(df: pd.DataFrame, class_column, class_values_list):
    """Ordena o dataset pelos valores da classe."""
    subsets_list = []
    for i in range(0, len(class_values_list)):
        subsets_list.append(df[df[class_column] == class_values_list[i]])

    df = subsets_list[0]
    for i in range(1, len(subsets_list)):
        df = df.append(subsets_list[i], ignore_index=True)

    return df

def print_folds_proportion(original_df: pd.DataFrame, folds_list, class_column, proportions=True):
    """Mostra a proporcao dos valores de classe do dataset original e de cada fold. normalize=True indica a proporcao,
    colocar normalize=False mostra a quantidade absoluta"""
    print('Original:')
    print(original_df[class_column].value_counts(normalize=proportions), '\n')

    print('Folds:')
    for i in range(0, len(folds_list)):
        print(i+1, ':')
        print('Tamanho:', len(folds_list[0]))
        print(folds_list[i][class_column].value_counts(normalize=proportions))
    return



if __name__=="__main__":
    filename = 'breast_cancer_original.csv'
    df = pd.read_csv(filename)
    class_column = 'class'
    k = 17

    folds_list = k_fold(df, k, class_column)
    print_folds_proportion(df, folds_list, class_column, False)


    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print_folds_proportion(df, folds_list, 'class')