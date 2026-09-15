"""
Application Streamlit — Prédiction du prix de vente d'une voiture
Conversion directe de l'application Gradio d'origine.

Lancement en local :  streamlit run app_voiture.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predict the selling price of a car",
    page_icon="🚗",
    layout="centered",
)

DESCRIPTION = (
    "This machine learning model allows us to predict the selling price of a car "
    "from the kms driven, present price, fuel type, seller type, transmission and "
    "age of the car."
)

# ----------------------------------------------------------------------
# Chargement des artefacts (mis en cache : chargés une seule fois)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    encoders = joblib.load("encoders.joblib")                # encodeurs
    pipe_from_grid = joblib.load("pipe_from_grid.joblib")    # pipeline du best modèle
    return encoders, pipe_from_grid


encoders, pipe_from_grid = load_artifacts()


# ----------------------------------------------------------------------
# Fonction de prédiction simple
# ----------------------------------------------------------------------
def Pred_func(Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age):
    # Encoder les valeurs des Fuel_Type, Seller_Type et Transmission
    Fuel_Type = encoders[0].transform([Fuel_Type])[0]
    Seller_Type = encoders[1].transform([Seller_Type])[0]
    Transmission = encoders[2].transform([Transmission])[0]
    # Vecteur des valeurs numériques
    x_new = np.array([Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age])
    x_new = x_new.reshape(1, -1)  # conversion en un tableau 2D
    # Prédire
    y_pred = pipe_from_grid.predict(x_new)
    # Arrondir
    y_pred = round(y_pred[0], 2)
    return str(y_pred) + "k$"


# ----------------------------------------------------------------------
# Fonction de prédiction multiple
# ----------------------------------------------------------------------
def Pred_func_csv(file):
    # Lire le fichier csv
    df = pd.read_csv(file)
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :].values:
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5])
        predictions.append(y_pred)

    df["Selling_Price"] = predictions
    return df


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
st.title("🚗 Car selling price prediction")

onglet1, onglet2 = st.tabs(["Simple Prediction", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Predict the selling price of a car with a single input")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            Kms_Driven = st.number_input("Kms Driven", value=0.0, step=1000.0, format="%.2f")
            Present_Price = st.number_input("Present Price", value=0.0, step=0.5, format="%.2f")
            Age = st.number_input("Age", value=0.0, step=1.0, format="%.2f")
        with col2:
            Fuel_Type = st.radio("Fuel Type", options=["Petrol", "Diesel", "CNG"])
            Seller_Type = st.radio("Seller Type", options=["Dealer", "Individual"])
            Transmission = st.radio("Transmission", options=["Manual", "Automatic"])

        soumettre = st.form_submit_button("Predict", type="primary")

    if soumettre:
        try:
            resultat = Pred_func(
                Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age
            )
            st.success(f"**Selling Price :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Predict the selling price of a car with multiple inputs")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age."
    )

    fichier = st.file_uploader("Upload a csv file", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Download a csv file",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
