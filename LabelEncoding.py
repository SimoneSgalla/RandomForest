import os.path

# Carica il dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump


data=[]
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

'''
    # Applica PCA per ridurre le dimensioni
    X = X.astype('float32')

    # Dividi il dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modello Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predizioni e valutazione
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Salva il modello addestrato
dump(model, 'random_forest_model.joblib')'''
