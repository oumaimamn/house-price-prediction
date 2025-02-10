import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Charger les données
data = pd.read_csv('housing.csv')

# Séparer les colonnes numériques et catégorielles
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=[object]).columns

# Remplir les valeurs manquantes pour les colonnes numériques avec la médiane
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Remplir les valeurs manquantes pour les colonnes catégorielles avec la valeur la plus fréquente
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encodage des variables catégorielles
label_encoder = {}
for col in categorical_columns:
    label_encoder[col] = LabelEncoder()
    data[col] = label_encoder[col].fit_transform(data[col])

# Diviser les données en X et y
X = data.drop('price', axis=1)
y = data['price']

# Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données (facultatif, mais peut aider)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner le modèle
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Interface utilisateur Streamlit
st.title("Prédiction des Prix de l'Immobilier")
st.write("Entrez les caractéristiques pour prédire le prix.")

# Entrées utilisateur
area = st.number_input("Surface (area)", 1000, 10000, 2000)
bedrooms = st.slider("Nombre de Chambres (bedrooms)", 1, 10, 3)
bathrooms = st.slider("Nombre de Salles de Bain (bathrooms)", 1, 5, 2)
stories = st.slider("Nombre d'Étages (stories)", 1, 5, 2)

# Utiliser les classes apprises par le LabelEncoder pour les colonnes catégorielles
mainroad = st.selectbox("Mainroad (yes/no)", label_encoder['mainroad'].classes_)
guestroom = st.selectbox("Guestroom (yes/no)", label_encoder['guestroom'].classes_)
basement = st.selectbox("Basement (yes/no)", label_encoder['basement'].classes_)
hotwaterheating = st.selectbox("Hotwaterheating (yes/no)", label_encoder['hotwaterheating'].classes_)
airconditioning = st.selectbox("Airconditioning (yes/no)", label_encoder['airconditioning'].classes_)
parking = st.slider("Nombre de Places de Parking (parking)", 1, 5, 2)
prefarea = st.selectbox("Prefarea (yes/no)", label_encoder['prefarea'].classes_)
furnishingstatus = st.selectbox("Furnishing Status", label_encoder['furnishingstatus'].classes_)

# Prédiction
if st.button("Prédire le Prix"):
    # Créer un tableau avec les entrées utilisateur
    input_data = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                            hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])

    # Appliquer l'encodage sur les colonnes catégorielles
    for i, col in enumerate(categorical_columns):
        # Vérifiez que la valeur appartient aux classes apprises
        if str(input_data[0, i + 4]) not in label_encoder[col].classes_:
            st.error(f"Valeur inconnue détectée pour la colonne {col}: {input_data[0, i + 4]}")
            st.stop()
        input_data[0, i + 4] = label_encoder[col].transform([input_data[0, i + 4]])[0]  # Encodage

    # Normalisation des données
    input_data = scaler.transform(input_data)

    # Prédire le prix avec le modèle
    prediction = model.predict(input_data)
    st.success(f"Le prix estimé est : ${prediction[0]:,.2f}")
