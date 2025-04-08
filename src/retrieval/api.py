import json
import os
from pathlib import Path

from flask import Flask, jsonify, request

from src.retrieval.embedding_factory import EmbeddingStoreFactory
from src.utils.local_logger import LocalLogger


def create_flask_app(
    config_path: str,
    logger: LocalLogger,
):
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
            logger.error("Query is required")
            return jsonify({"error": "Query is required"}), 400

        query = data["query"]
        n_results = data.get("n_results")

        try:
            results = embedding_store.search(query, n_results)
            logger.info(f"Search results: {results}")
            return jsonify({"results": results})
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/update", methods=["POST"])
    def update():
        data = request.json
        if not data or "document" not in data:
            logger.error("Document is required")
            return jsonify({"error": "Document is required"}), 400

        try:
            metadata = data.get("metadata", {})
            document = data["document"]
            success = embedding_store.update(document, metadata)
            if success:
                logger.info("Update successful")
                return jsonify({"status": "success"})
            else:
                logger.error("Updates not allowed")
                return jsonify({"error": "Updates not allowed"}), 403
        except Exception as e:
            logger.error(f"Error updating: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health", methods=["GET"])
    def health():
        try:
            status = embedding_store.health_check()
            logger.info("Health check passed")
            return jsonify(status)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500

    return app
