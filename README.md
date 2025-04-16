
Anomaly Detection System

This project is a Flask-based anomaly detection system that uses machine learning to identify suspicious activities and stores them in a Neo4j database.

## Features

- REST API with Flask
- Anomaly detection with Gradient Boosting
- Data storage in Neo4j
- Data preprocessing
- Variable encoding
  
## Configuration Requise

- Python 3.11+
- Flask
- pandas
- numpy
- joblib
- neo4j
- py2neo

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Saadia-Akerkouch/Insider-Threat-Detection.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py` : Flask application entry point
- `gradient_boosting_model.pkl` : Trained ML Model
- `minmax_scaler.pkl` :Scaler for normalization
- `label_encoders.pkl` : Encoders for categorical variables
  
## Utilization

The API accepts POST requests to the root endpoint ('/') with logs in JSON format.

Exemple de requête :
```bash
curl -X POST http://localhost:5000 -H "Content-Type: application/json" -d '[{
    "timestamp": "2024-10-12T14:45:00",
  "date": "2024-10-12",
  "time": "14:45:00",
  "hour": 14,
  "weekday": "Saturday",
  "user_id": 1582,
  "ip_address": "198.162.1.101",
  "ip1": 192,
  "ip2": 168,
  "ip3": 1,
  "ip4": 101,
  "activity_type": "File Access",
  "activity_group": "Confidential",
  "resource_accessed": "/private/hr/employee_records.pdf",
  "file_name": "employee_records.pdf",
  "file_size": 300.5,
  "is_large_file": 1,
  "login_attempts": 1,
  "action": "Download"
}]'
```

## Configuration Neo4j

The connection to Neo4j is configured with the following variables in `main.py` :
- NEO4J_URI
- NEO4J_USER
- NEO4J_PASSWORD

## Deployment

The project is configured to be deployed on Replit.

## License

All rights reserved.
