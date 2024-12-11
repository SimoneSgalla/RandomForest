import ipaddress
import os.path

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carica il dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import load
from sklearn.feature_selection import VarianceThreshold
from joblib import dump
from sklearn.preprocessing import TargetEncoder


data = []
for i in range(22):
    data.append(pd.read_csv('../Dataset2/Network_dataset_'+str(1+i)+'.csv', low_memory=False))
df = pd.concat(data)

X = df.drop('label', axis=1, errors='ignore')
X = X.drop('type', axis=1, errors='ignore')
y = df['type']

    # Applica Label Encoding su eventuali colonne categoriche rimanenti

    # Salva i trasformatori
enc_auto = TargetEncoder(smooth="auto")
X_trans = enc_auto.fit_transform(X, y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

dump(X_trans, 'random_forest_model.joblib')


