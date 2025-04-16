from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from py2neo import Graph, Node, Relationship


app = Flask(__name__)

# ==========  Chargement des modèles & outils ==========
model = joblib.load("gradient_boosting_model.pkl")
#model = joblib.load("decision_tree_model (2).pkl")
scaler = joblib.load("minmax_scaler (1).pkl")
label_encoders = joblib.load(
    "label_encoders (3).pkl")  


# ========== Encodage =====================
def encode_value(col_name, value):
    """Encode une valeur selon les mappings sauvegardés."""
    return label_encoders[col_name].get(str(value), -1)


# =========== Pré-traitement ================================
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

    log['is_night_time'] = 1 if log.get('hour', 0) < 6 or log.get(
        'hour', 0) > 20 else 0

    # Créer un DataFrame
    df = pd.DataFrame([log])

    # Colonnes numériques à normaliser
    num_cols = ['file_size', 'login_attempts']
    df[num_cols] = scaler.transform(df[num_cols])

    # Sélectionner seulement les features utilisés pendant l'entraînement
    df = df[list(model.feature_names_in_)]  # très important

    return df


# ================== Connexion à la BD ========================
NEO4J_URI = "neo4j+s://8f4266b6.databases.neo4j.io" 
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "3N3pP_mAhZxdT9J_GY9diTO9vS2bSs8-flH6M0JuH8A"

graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    graph.run("RETURN 1").evaluate()
    print("Connexion réussie à Neo4j Aura")
except Exception as e:
    print("Échec de la connexion :", e)


# ========== Fonction d'insertion dans Neo4j ===============
def insert_log_into_neo4j(log):
    try:
        user = Node("User", user_id=log["user_id"])
        ip = Node("IPAddress", address=log["ip_address"])

        # Clé composite pour éviter les doublons
        resource_key = f"{log['resource_accessed']}|{log['file_name']}"
        resource = Node("Resource", id=resource_key, path=log["resource_accessed"], file_name=log["file_name"])

        activity = Node(
            "Activity",
            activity_type=log["activity_type"],
            activity_group=log["activity_group"],
            action=log["action"],
            file_size=log["file_size"],
            is_large_file=log["is_large_file"],
            login_attempts=log["login_attempts"],
            is_anomaly=log["is_anomaly"],
            date=log["date"],
            time=log["time"],
            weekday=log["weekday"],
            hour=log["hour"]
        )

        # Créer les relations
        rel1 = Relationship(user, "PERFORMED", activity)
        rel2 = Relationship(activity, "TARGETED", resource)
        rel3 = Relationship(activity, "USED_IP", ip)
        # Créer le noeud AnomalyStatus
        anomaly_status = Node("AnomalyStatus", status="anomaly" if log["is_anomaly"] else "normal")

        # Relation entre Activity et AnomalyStatus
        rel4 = Relationship(activity, "HAS_STATUS", anomaly_status)

        # Merge et création
        graph.merge(anomaly_status, "AnomalyStatus", "status")
        graph.create(rel4)


        # Centralisation
        graph.merge(user, "User", "user_id")
        graph.merge(ip, "IPAddress", "address")
        graph.merge(resource, "Resource", "id")
        graph.create(activity)
        graph.create(rel1)
        graph.create(rel2)
        graph.create(rel3)

        print("Log inséré avec succès dans Neo4j !")

    except Exception as e:
        print("[ERREUR] Insertion dans Neo4j a échoué :", str(e))

# ========== Endpoint de prédiction ==========
@app.route('/', methods=['POST'])
def predict():
    try:
        raw_log = request.get_json()[0]
        processed_df = preprocess_input(raw_log)
        print("processed_df:", processed_df)

        prediction = model.predict(processed_df)[0]
        print("prediction:", prediction)
        anomaly_score = model.predict_proba(processed_df)[0][1]

        result = raw_log.copy()
        result['is_anomaly'] = bool(prediction)

        # Sauvegarder le log dans Neo4j
        insert_log_into_neo4j(result)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Get the PORT environment variable (default to 5000 for local testing)
    port = int(os.environ.get("PORT", 5000))
    # Bind to all interfaces and use the correct port
    app.run(host='0.0.0.0', port=port)
