import pandas as pd
import math

# class BranchAndBound:
#
#     def __init__(self, dataset):
#         self.dataset = dataset


def branch_and_bound(df: pd.DataFrame, class_name, target_feature_number, function, best_measurement, maximize_measurement=True):
    """
    Algoritmo branch and bound recursivo.
    :param df: o dataset de interesse.
    :param class_name: o nome do campo de classe.
    :param target_feature_number: o numero de feaures desejado para o subset.
    :param function: a funcao de avaliazao. Ela sera executada como function(df, class_name)
    :param best_measurement: a melhor medida. Na primeira iteracao, usar uma medida apropriada
    :param maximize_measurement: se a medida deve ser maximixada. Portanto, se falso, ele retornara o subset com menor medida.
    :return: data_subset, measuremente ou falso
    """

    # mata os warnings
    # pd.options.mode.chained_assignment = None  # default='warn'

    measurement = function(df, class_name)

    if maximize_measurement and measurement <= best_measurement:
        return False
    if not maximize_measurement and measurement >= best_measurement:
        return True

    # pega uma lista de features sem a classe
    features_list = list(df.columns.values)
    features_list.remove(class_name)

    # seguranca
    if len(features_list) < target_feature_number:
        raise Exception('Inserir target_feature_number maior do que o numero de features atual.')

    # caso base: retorna o dataframe (ja reduzido) e a medida melhor
    if len(features_list) == target_feature_number:
        return df, measurement

    best_df = None
    for feature in features_list:
        next_df = df.drop(feature, axis=1)
        result = branch_and_bound(next_df, class_name, target_feature_number, function, best_measurement, maximize_measurement)
        if result:
            best_df, best_measurement = result

    if best_df is None:
        return False
    return best_df, best_measurement


    pass

def centroid_distance(df: pd.DataFrame, class_name):
    # separa os grupos de classe 0 e 1
    group_0 = df[df[class_name] == 0]
    group_1 = df[df[class_name] == 1]

    # retira os campos classe para nao computar distancia entre eles
    group_0 = group_0.drop(class_name, axis=1)
    group_1 = group_1.drop(class_name, axis=1)

    # centroide as medias de cada feature
    centroid_0 = []
    for i in group_0:
        centroid_0.append(group_0[i].mean())
    centroid_1 = []
    for i in group_1:
        centroid_1.append(group_1[i].mean())

    distance = euclidian_distance(centroid_0, centroid_1)
    return distance

def euclidian_distance(arr1, arr2):
    if len(arr1) != len(arr2):
        raise Exception('Euclidian distance: arrays of differente lenghts!')

    distance = 0
    for i in range(0, len(arr1)):
        distance += (arr1[i] - arr2[i])**2
    return math.sqrt(distance)


