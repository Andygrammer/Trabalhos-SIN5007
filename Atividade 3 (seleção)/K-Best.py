
# coding: utf-8

# In[ ]:


from sklearn.feature_selection import  VarianceThreshold,SelectKBest,f_classif

features = ['class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','irradiat']
filename = 'dataset-full.csv'
breast_cancer_dataset = pd.read_csv(filename, names=features)
y = breast_cancer_dataset['class']
breast_cancer_dataset = breast_cancer_dataset.drop('class', 1)
X = breast_cancer_dataset

selector=SelectKBest(score_func=f_classif,k=6)
selector.fit(X,y)
print(selector.get_support(indices=True))

