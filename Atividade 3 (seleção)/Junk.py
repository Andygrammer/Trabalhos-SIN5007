import pandas as pd

filename = 'breast_cancer_final.csv'
breast_cancer_dataset = pd.read_csv(filename)

class_name = 'class'

class_0 = breast_cancer_dataset[breast_cancer_dataset[class_name]==0]
class_none = class_0.drop(class_name, axis=1, inplace=True)

print(class_0)
print(breast_cancer_dataset)
print(class_none)
gp = breast_cancer_dataset.groupby(class_name)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(class_0)
    # print(class_00)
    # print(gp.get_group(name=0))
    print('end')

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(breast_cancer_dataset)