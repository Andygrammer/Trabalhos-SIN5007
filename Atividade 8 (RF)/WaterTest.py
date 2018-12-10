import h2o

h2o.init()
h2o.remove_all()

from h2o.estimators.random_forest import H2ORandomForestEstimator

covtype_df = h2o.import_file('breast_cancer_mod.csv')
# deixa no-rec como negativo
covtype_df['class'] = covtype_df['class'].relevel('no-recurrence-events')

# mtries eh automatico sqrt(n)
rf_v1 = H2ORandomForestEstimator(
    ntrees=500,
    score_each_iteration=True,
    nfolds=17,
    mtries=8
)

rf_v1.train(y='class', training_frame=covtype_df)


# ### Model Construction
# H2O in Python is designed to be very similar in look and feel to to scikit-learn. Models are initialized individually with desired or default parameters and then trained on data.
#
# **Note that the below example uses model.train() as opposed the traditional model.fit()**
# This is because h2o-py takes column indices for the feature and response columns AND the whole data frame, while scikit-learn takes in a feature frame and a response frame.
#
# H2O supports model.fit() so that it can be incorporated into a scikit-learn pipeline, but we advise using train() in all other cases.

# In[ ]:

print('PRINT')

print(rf_v1)
threshold = rf_v1.find_threshold_by_max_metric('accuracy')
print(threshold)
print(rf_v1.confusion_matrix(['accuracy'], thresholds=[threshold]))
print(rf_v1.accuracy(thresholds = [threshold]))
print(rf_v1.precision(thresholds=[threshold]))
print(rf_v1.recall(thresholds=[threshold]))

h2o.remove_all()