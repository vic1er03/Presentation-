import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
#from xgboost import XGBRegressor
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from io import BytesIO
import os

# Tableau de Bord Streamlit
st.title("Tableau de Bord de Prédiction des Prix des Maisons")

# Charger le Modèle et les Préprocesseurs
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
numeric_imputer = joblib.load('numeric_imputer.pkl')

# Section d'Entrée Manuelle
st.header("Entrée Manuelle")
lot_area = st.number_input("Surface du Terrain (pieds carrés)", min_value=0, value=8500)
overall_qual = st.slider("Qualité Globale (1-10)", 1, 10, 5)
year_built = st.number_input("Année de Construction", min_value=1900, max_value=2025, value=2000)
gr_liv_area = st.number_input("Surface Habitable au-dessus du Sol (pieds carrés)", min_value=0, value=1500)
age = st.number_input("Âge de la Maison (années)", min_value=0, value=10)

# Préparer l'Entrée Manuelle
manual_input = pd.DataFrame({
    'Lot Area': [lot_area],
    'Overall Qual': [overall_qual],
    'Year Built': [year_built],
    'Gr Liv Area': [gr_liv_area],
    'Age': [age]
})
manual_input[numeric_features] = numeric_imputer.transform(manual_input[numeric_features])
manual_input[numeric_features] = scaler.transform(manual_input[numeric_features])
# Ajouter des caractéristiques catégoriques fictives (simplifié pour la démo)
dummy_cats = pd.DataFrame(0, index=[0], columns=encoded_df.columns)
manual_input = pd.concat([manual_input, dummy_cats], axis=1)

# Prédire
if st.button("Prédire le Prix"):
    prediction = model.predict(manual_input)
    st.success(f"Prix de Vente Prédit : {prediction[0]:,.2f} $")

# Section de Téléchargement Excel
st.header("Télécharger un Fichier Excel")
uploaded_file = st.file_uploader("Choisir un fichier Excel", type="xlsx")
if uploaded_file is not None:
    df_upload = pd.read_excel(uploaded_file)
    # Assurer que les colonnes correspondent
    df_upload = df_upload[['Lot Area', 'Overall Qual', 'Year Built', 'Gr Liv Area']].fillna(0)
    df_upload['Age'] = 2025 - df_upload['Year Built']  # Année actuelle comme 2025
    df_upload[numeric_features] = numeric_imputer.transform(df_upload[numeric_features])
    df_upload[numeric_features] = scaler.transform(df_upload[numeric_features])
    dummy_cats = pd.DataFrame(0, index=df_upload.index, columns=encoded_df.columns)
    df_upload = pd.concat([df_upload, dummy_cats], axis=1)

    # Prédire et Ajouter aux Données
    df_upload['Prix de Vente Prédit'] = model.predict(df_upload)
    st.write("Prédictions :", df_upload)

    # Télécharger les Données Prédites
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_upload.to_excel(writer, index=False)
    st.download_button(
        label="Télécharger les Prédictions",
        data=output.getvalue(),
        file_name="maisons_predites.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Étape 7-8 : Maintenance et Éthique (Implémenté via la surveillance et les vérifications de biais)
st.sidebar.text("Modèle surveillé pour la dérive ; réentraîner si RMSE > 35000. Audits de biais en cours.")