# Carica il dataset
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.preprocessing import TargetEncoder

data = []
data2 = []
for i in range(22):
    data.append(pd.read_csv('../Dataset2/Network_dataset_'+str(1+i)+'.csv', low_memory=False, usecols=['src_ip']))
    data.append(pd.read_csv('../Dataset2/Network_dataset_' + str(1 + i) + '.csv', low_memory=False, usecols=['label']))
df = pd.concat(data)
df2 = pd.concat(data2)

X = df
y = df2

    # Applica Label Encoding su eventuali colonne categoriche rimanenti

    # Salva i trasformatori
enc_auto = TargetEncoder(smooth="auto")
X_trans = enc_auto.fit_transform(X, y)

dump(X_trans, 'random_forest_model.joblib')



