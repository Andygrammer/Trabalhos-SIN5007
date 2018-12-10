import pandas as pd
import BranchAndBound

# 36 features
filename = 'breast_cancer_final.csv'
breast_cancer_dataset = pd.read_csv(filename)

class_name = 'class'
target_number = 34

df, measeurement = BranchAndBound.branch_and_bound(
    df=breast_cancer_dataset,
    class_name=class_name,
    target_feature_number=target_number,
    function=BranchAndBound.centroid_distance,
    best_measurement=0
)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)

original_features = list(breast_cancer_dataset.columns.values)
selected_features = list(df)
removed = []
for i in original_features:
    if i not in selected_features:
        removed.append(i)
print(removed)
