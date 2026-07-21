from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from google.cloud import bigquery
import asyncio
import json
import os
from dotenv import load_dotenv
import traceback
import vertexai
from vertexai import agent_engines
import sys

load_dotenv()

app = Flask(__name__)

GCP_PROJECT = os.getenv("GCP_PROJECT")
AGENT_RESOURCE = "projects/512322842999/locations/us-central1/reasoningEngines/999058596194942976"
REQUIRE_IAP = os.getenv("REQUIRE_IAP", "").lower() in ("1", "true", "yes")

bq = bigquery.Client(project=GCP_PROJECT)


def current_user():
    raw = request.headers.get("X-Goog-Authenticated-User-Email", "")
    return raw.replace("accounts.google.com:", "")


@app.before_request
def _enforce_auth():
    if REQUIRE_IAP and request.path.startswith("/api/") and not current_user():
        return jsonify({"error": "Unauthorized"}), 401


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


def _shape_event(event):
    author = event.get("author", "?")
    out = []
    for part in (event.get("content") or {}).get("parts", []) or []:
        if part.get("text"):
            out.append({"author": author, "text": part["text"]})
        elif "function_call" in part:
            out.append({"author": author, "call": part["function_call"]["name"]})
        elif "function_response" in part:
            out.append({"author": author, "response": part["function_response"]["name"]})
    if event.get("error_code"):
        out.append({"author": author, "error": event.get("error_message") or event["error_code"]})
    return out


@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.get_json(silent=True) or {}
    dataset     = data.get("dataset")
    table       = data.get("table")
    action      = data.get("action")
    description = data.get("description")

    if not all([dataset, table, action, description]):
        return jsonify({"error": "Tous les champs sont requis."}), 400

    spec = f"{action} dans la table {dataset}.{table} : {description}"
    user_id = current_user() or "front-user"

    try:
        vertexai.init(project=GCP_PROJECT, location="us-central1")
        agent = agent_engines.get(AGENT_RESOURCE)
        session = agent.create_session(user_id=user_id, state={"spec": spec})
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500

    def sse(payload):
        return f"data: {json.dumps(payload)}\n\n"

    def generate():
        loop = asyncio.new_event_loop()
        stream = agent.async_stream_query(
            user_id=user_id,
            session_id=session["id"],
            message="Traite la spec",
        )
        yield sse({"status": "started", "spec": spec})
        try:
            while True:
                try:
                    event = loop.run_until_complete(stream.__anext__())
                except StopAsyncIteration:
                    break
                for item in _shape_event(event):
                    yield sse(item)
            yield sse({
                "status": "done",
                "message": f"Workflow terminé — {action} sur {dataset}.{table}",
            })
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            yield sse({"status": "error", "error": str(e)})
        finally:
            try:
                loop.run_until_complete(stream.aclose())
            except Exception:
                pass
            loop.close()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)
