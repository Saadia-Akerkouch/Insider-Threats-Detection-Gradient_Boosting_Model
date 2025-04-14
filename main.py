from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from neo4j import GraphDatabase

app = Flask(__name__)

# ==========  Chargement des modèles & outils ==========
model = joblib.load("gradient_boosting_model.pkl")
#model = joblib.load("decision_tree_model (2).pkl")
scaler = joblib.load("minmax_scaler (1).pkl")
label_encoders = joblib.load("label_encoders (3).pkl")  # dict : {col: {str: int}}

# ========== Décodage / Encodage ==========
def encode_value(col_name, value):
    """Encode une valeur selon les mappings sauvegardés."""
    return label_encoders[col_name].get(str(value), -1)


def decode_value(col_name, value):
    """Décode une valeur encodée selon les mappings sauvegardés."""
    mapping = label_encoders[col_name]
    inv_map = {v: k for k, v in mapping.items()}
    return inv_map.get(value, value)

# ========== Fonction utilitaire ==========
def preprocess_input(log):
    log = log.copy()

    # Encodage catégoriel
    for col in ['weekday', 'activity_type', 'activity_group', 'action']:
        log[col] = encode_value(col, log.get(col, ""))

    # Ajouter les features si absents
    log['is_weekend'] = 1 if log['weekday'] in [
        encode_value('weekday', 'Saturday'),
        encode_value('weekday', 'Sunday')
    ] else 0

    log['is_night_time'] = 1 if log.get('hour', 0) < 6 or log.get('hour', 0) > 20 else 0

    # Créer un DataFrame
    df = pd.DataFrame([log])

    # Colonnes numériques à normaliser
    num_cols = ['file_size', 'login_attempts']
    df[num_cols] = scaler.transform(df[num_cols])

    # Sélectionner seulement les features utilisés pendant l'entraînement
    df = df[list(model.feature_names_in_)]  # très important

    return df




# ========== Endpoint de prédiction ==========
@app.route('/', methods=['POST'])
def predict():
    try:
        raw_log = request.get_json()[0]
        processed_df = preprocess_input(raw_log)
        print("processed_df:",processed_df)

        prediction = model.predict(processed_df)[0]
        print("prediction:",prediction)
        anomaly_score = model.predict_proba(processed_df)[0][1]

        result = raw_log.copy()
        result['is_anomaly'] = bool(prediction)
        

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400





if __name__ == '__main__':
    # Get the PORT environment variable (default to 5000 for local testing)
    port = int(os.environ.get("PORT", 5000))
    # Bind to all interfaces and use the correct port
    app.run(host='0.0.0.0', port=port)
