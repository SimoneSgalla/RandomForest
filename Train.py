from joblib import load
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump


# Carica il modello e i trasformatori salvati
model = load('random_forest_model.joblib')
label_encoders = load('label_encoders.joblib')
column_names = pd.read_csv('feature_columns.csv', header=None).squeeze().tolist()

# Carica il nuovo dataset
new_df = pd.read_csv('../Dataset2/Network_dataset_2.csv', low_memory=False)

# Seleziona le colonne necessarie e riordina
target_column = 'label'  # Cambia con la tua colonna target
y_new = new_df[target_column]
new_df = new_df.drop('label', axis=1, errors='ignore')
X_new = new_df[column_names]

# Applica Label Encoding alle colonne categoriche
for col in X_new.select_dtypes(include=['object']).columns:
    if col in label_encoders:  # Usa l'encoder salvato se disponibile
        X_new[col] = label_encoders[col].transform(X_new[col].astype(str))
    else:  # Crea un nuovo encoder per nuove colonne
        le = LabelEncoder()
        X_new[col] = le.fit_transform(X_new[col].astype(str))
        label_encoders[col] = le  # Aggiorna i trasformatori

if 'ip_address' in X_new.columns:
    ip_features = X_new['ip_address'].apply(ip_to_features).tolist()
    ip_df = pd.DataFrame(ip_features, columns=['ip_part1', 'ip_part2', 'ip_part3', 'ip_part4'])
    X_new = pd.concat([X_new.drop('ip_address', axis=1), ip_df], axis=1)

X_new = X_new.astype('float32')

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Riaddestra il modello
model.fit(X_train_new, y_train_new)

# Valutazione
y_pred_new = model.predict(X_test_new)
print(f'Accuracy: {accuracy_score(y_test_new, y_pred_new)}')
print(classification_report(y_test_new, y_pred_new))

# Salva il modello aggiornato

dump(model, 'random_forest_model_updated.joblib')
