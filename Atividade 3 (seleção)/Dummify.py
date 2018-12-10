import pandas as pd
import Normalizer


def substitute_in_interval(df, feat_name, beg, end, interval):
    it = 0
    l = beg
    r = beg+interval-1



    while r <= end:
        interval_value = '{}-{}'.format(l, r)

        df[feat_name] = df[feat_name].replace(interval_value, it)
        it += 1
        l += interval
        r += interval

    return df


breast_cancer_filename = 'breast_cancer_original.csv'
breast_cancer_dataset = pd.read_csv(breast_cancer_filename)



# transformar binarios em 0-1
breast_cancer_dataset['class'] = breast_cancer_dataset['class'].replace('no-recurrence-events', 0)
breast_cancer_dataset['class'] = breast_cancer_dataset['class'].replace('recurrence-events', 1)

breast_cancer_dataset['node-caps'] = breast_cancer_dataset['node-caps'].replace('no', 0)
breast_cancer_dataset['node-caps'] = breast_cancer_dataset['node-caps'].replace('yes', 1)

breast_cancer_dataset['breast'] = breast_cancer_dataset['breast'].replace('left', 0)
breast_cancer_dataset['breast'] = breast_cancer_dataset['breast'].replace('right', 1)

breast_cancer_dataset['irradiat'] = breast_cancer_dataset['irradiat'].replace('no', 0)
breast_cancer_dataset['irradiat'] = breast_cancer_dataset['irradiat'].replace('yes', 1)

# tratar missing values usando moda
breast_cancer_dataset['node-caps'] = breast_cancer_dataset['node-caps'].replace('?', breast_cancer_dataset['node-caps'].value_counts().idxmax())
breast_cancer_dataset['breast-quad'] = breast_cancer_dataset['breast-quad'].replace('?', breast_cancer_dataset['breast-quad'].value_counts().idxmax())


breast_cancer_dataset = substitute_in_interval(breast_cancer_dataset, 'age', 10, 99, 10)
breast_cancer_dataset = substitute_in_interval(breast_cancer_dataset, 'tumor-size', 0, 59, 5)
breast_cancer_dataset = substitute_in_interval(breast_cancer_dataset, 'inv-nodes', 0, 39, 3)


# breast_cancer_dataset['inv_nodes'] = breast_cancer_dataset['inv_nodes'].replace('?', breast_cancer_dataset['breast-quad'].value_counts().idxmax())



# colocar dummies
# breast_cancer_dataset = pd.get_dummies(breast_cancer_dataset)

# retirar breast-quad e tratar menopausa
breast_cancer_dataset = breast_cancer_dataset.drop('breast-quad', axis=1)
breast_cancer_dataset['menopause'] = breast_cancer_dataset['menopause'].replace('lt40', 1)
breast_cancer_dataset['menopause'] = breast_cancer_dataset['menopause'].replace('ge40', 1)
breast_cancer_dataset['menopause'] = breast_cancer_dataset['menopause'].replace('premeno', 0)

attributes = list(breast_cancer_dataset.columns.values)
attributes.remove('class')
print(attributes)
breast_cancer_dataset = Normalizer.normalize_by_zscore(breast_cancer_dataset, attributes)


output_filename = 'breast_cancer_final_no_meno_quad.csv'
breast_cancer_dataset.to_csv(output_filename, index=False)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(breast_cancer_dataset)

