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


def testOnNewDataset(model, path):
    # Carica il nuovo dataset
    new_data = pd.read_csv(path, low_memory=False)

    # Pre-elaborazione (ad esempio, encoding delle colonne categoriche)
    for col in new_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        new_data[col] = le.fit_transform(new_data[col])

    # Separare le caratteristiche
    X_new = new_data.drop('label', axis=1, errors='ignore')  # Rimuove 'label' se presente

    # Ridurre dimensionalità con PCA (se applicabile)
    X_new = X_new.astype('float32')

    # Predizioni sul nuovo dataset
    #predictions = model.predict(X_new)

    # Valutazione se c'è una colonna target
    y_new = new_data['label']
    X_train_new, X_test_new, y_train_new, y_test_new = \
        (train_test_split(X_new, y_new, test_size=0.2,random_state=42))
    model.fit(X_train_new, y_train_new)
    y_pred_new = model.predict(X_test_new)
    print(f'Accuracy: {accuracy_score(y_test_new, y_pred_new)}')
    print(classification_report(y_test_new, y_pred_new))

    dump(model, 'random_forest_model_updated'+str(18+i)+'.joblib')

    # Se esiste una colonna target, calcola l'accuracy e stampa il report
'''    if 'label' in new_data.columns:
        y_new = new_data['label']
        y_new = y_new[X_new.index]
        print(f"Accuracy sul nuovo dataset: {accuracy_score(y_new, predictions)}")
        print(classification_report(y_new, predictions))
    else:
        print("Predizioni sul nuovo dataset:")
        print(predictions)'''



file_path = 'random_forest_model.joblib'
if os.path.exists(file_path):
    model = load('random_forest_model.joblib')
    for i in range(5):
        testOnNewDataset(model, '../Dataset2/Network_dataset_'+str(19+i)+'.csv')
else:
    # Carica il dataset
    df = pd.read_csv('../Dataset2/Network_dataset_1.csv', low_memory=False)

    # Usa il nome corretto della colonna target
    target_column = 'label'  # Sostituisci con il nome corretto della colonna target

    print(df.columns)

    # Encoding per variabili categoriche
    X = df.drop(target_column, axis=1, errors='ignore')
    print(X.columns)
    y = df[target_column]

    # Applica Label Encoding su eventuali colonne categoriche rimanenti
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Salva i trasformatori
    dump(label_encoders, 'label_encoders.joblib')

    # Salva i nomi delle colonne dopo il preprocessing completo
    column_names = X.columns
    pd.Series(column_names).to_csv('feature_columns.csv', index=False, header=False)

    # Riduci il dataset (opzionale)
    #X = X.sample(frac=0.1, random_state=42)
    #y = y[X.index]

    # Applica PCA per ridurre le dimensioni
    X = X.astype('float32')
    #pca = PCA(n_components=100)
    #X_reduced = pca.fit_transform(X)

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
    dump(model, 'random_forest_model.joblib')





