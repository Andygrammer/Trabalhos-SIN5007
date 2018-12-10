import pandas as pd

df = pd.read_excel("dsFullComMedias.xlsx")

# tratar missing values usando moda
df['node-caps'] = df['node-caps'].replace('?', df['node-caps'].value_counts().idxmax())
df['breast-quad'] = df['breast-quad'].replace('?', df['breast-quad'].value_counts().idxmax())



df['node-caps'] = df['node-caps'].replace('no', 0)
df['node-caps'] = df['node-caps'].replace('yes', 1)

df['breast'] = df['breast'].replace('left', 0)
df['breast'] = df['breast'].replace('right', 1)

df['irradiat'] = df['irradiat'].replace('no', 0)
df['irradiat'] = df['irradiat'].replace('yes', 1)

df['class'] = df['class'].replace('no-recurrence-events', 0)
df['class'] = df['class'].replace('recurrence-events', 1)

#transforma os atributos categóricos em dummies
df = pd.get_dummies(df, columns=["menopause", "breast-quad"])


# df = df.drop('breast-quad', axis=1)
# df['menopause'] = df['menopause'].replace('lt40', 1)
# df['menopause'] = df['menopause'].replace('ge40', 1)
# df['menopause'] = df['menopause'].replace('premeno', 0)

df.to_csv('dataset_dummies.csv', index=False)
