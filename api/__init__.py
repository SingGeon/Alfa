"""Flask application factory."""
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import config

# The React frontend (frontend/web) builds into frontend/dist - see the
# README's "Frontend" section. Flask serves that build directly, so a single
# `python run_api.py` is still all you need in production.
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app() -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(
        __name__, static_folder=str(DIST_DIR), static_url_path=""
    )
    app.config["JSON_SORT_KEYS"] = False

    from api.routes import api_bp

    from evaluation.api import evaluation_bp

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(evaluation_bp, url_prefix="/api/evaluation")

    def _spa_index():
        if not (DIST_DIR / "index.html").exists():
            return (
                "Frontend not built. Run: cd frontend/web && npm install && npm run build",
                503,
                {"Content-Type": "text/plain; charset=utf-8"},
            )
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/")
    def index():
        return _spa_index()

    # Client-side routes (/dashboard, /scout, /asset/..., old *.html links)
    # all resolve to the SPA shell; unknown /api paths stay JSON 404s.
    @app.errorhandler(404)
    def not_found(exc):
        if request.path.startswith("/api/") or request.method != "GET":
            return jsonify({"error": "Not found"}), 404
        return _spa_index()

    return app
