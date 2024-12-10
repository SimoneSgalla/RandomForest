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


data = []
for i in range(22):
    data.append(pd.read_csv('../Dataset2/Network_dataset_1.csv', low_memory=False))
df = pd.concat(data)

    # Applica Label Encoding su eventuali colonne categoriche rimanenti

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

    # Salva i trasformatori
dump(label_encoders, 'label_encoders.joblib')

    # Salva i nomi delle colonne dopo il preprocessing completo
column_names = df.columns
pd.Series(column_names).to_csv('feature_columns.csv', index=False, header=False)


