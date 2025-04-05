from flask import Flask, request, jsonify
import os
import json
from typing import Dict, Any

from retrieval.embedding_factory import EmbeddingStoreFactory


def create_flask_app(config_path: str):
    """Create and configure the Flask application with API endpoints.

    Args:
        config_path: Path to the configuration file.

    Returns:
        A configured Flask application
    """
    app = Flask(__name__)

    # Load config and initialize store
    if config_path is None:
        config_path = os.environ.get("EMBEDDING_CONFIG", "embedding_config.json")

    # Load the configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create the embedding store
    embedding_store = EmbeddingStoreFactory.create_store(config)

    @app.route("/api/search", methods=["POST"])
    def search():
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Query is required"}), 400

        query = data["query"]
        n_results = data.get("n_results")

        try:
            results = embedding_store.search(query, n_results)
            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/update", methods=["POST"])
    def update():
        data = request.json
        if not data or "document" not in data:
            return jsonify({"error": "Document is required"}), 400

        document = data["document"]
        metadata = data.get("metadata", {})

        try:
            success = embedding_store.update(document, metadata)
            if success:
                return jsonify({"status": "success"})
            else:
                return jsonify({"error": "Updates not allowed"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health", methods=["GET"])
    def health():
        try:
            status = embedding_store.health_check()
            return jsonify(status)
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    return app
