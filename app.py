from flask import Flask, render_template, request, jsonify
import google.auth
import google.auth.transport.requests
from google.cloud import bigquery
import json
import os
from dotenv import load_dotenv
import requests
import traceback
import vertexai
from vertexai import agent_engines
import sys

load_dotenv()

app = Flask(__name__)

GCP_PROJECT = os.getenv("GCP_PROJECT")

bq = bigquery.Client(project=GCP_PROJECT)

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/datasets")
def get_datasets():
    try:
        datasets = [
            {"id": d.dataset_id, "location": getattr(d, "location", "—")}
            for d in bq.list_datasets()
        ]
        return jsonify(datasets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tables/<dataset_id>")
def get_tables(dataset_id):
    try:
        tables = [t.table_id for t in bq.list_tables(dataset_id)]
        return jsonify(tables)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/preview/<dataset_id>/<table_id>")
def get_preview(dataset_id, table_id):
    try:
        table_ref = f"{GCP_PROJECT}.{dataset_id}.{table_id}"
        query = f"SELECT * FROM `{table_ref}` LIMIT 5"
        results = bq.query(query).result()

        schema = [field.name for field in results.schema]
        rows = [[str(val) if val is not None else "null" for val in row] for row in results]

        return jsonify({"schema": schema, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




AGENT_RESOURCE = "projects/512322842999/locations/us-central1/reasoningEngines/999058596194942976"

@app.route("/api/submit", methods=["POST"])
def submit():
    try:
        data = request.get_json()
        dataset     = data.get("dataset")
        table       = data.get("table")
        action      = data.get("action")
        description = data.get("description")

        if not all([dataset, table, action, description]):
            return jsonify({"error": "Tous les champs sont requis."}), 400

        spec = f"{action} dans la table {dataset}.{table} : {description}"

        vertexai.init(project=GCP_PROJECT, location="us-central1")
        agent = agent_engines.get(AGENT_RESOURCE)
        session = agent.create_session(user_id="front-user", state={"spec": spec})
        print (spec)

        events = []
        for event in agent.stream_query(
            user_id="front-user",
            session_id=session["id"],
            message="Traite la spec",
        ):
            author = event.get("author", "?")
            for part in (event.get("content") or {}).get("parts", []) or []:
                if part.get("text"):
                    events.append({"author": author, "text": part["text"]})
                elif "function_call" in part:
                    fc = part["function_call"]
                    events.append({"author": author, "call": fc["name"]})
                elif "function_response" in part:
                    fr = part["function_response"]
                    events.append({"author": author, "response": fr["name"]})

        return jsonify({
            "status": "success",
            "message": f"Workflow terminé — {action} sur {dataset}.{table}",
            "events": events
        })

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)