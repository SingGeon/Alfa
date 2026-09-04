"""Flask application factory."""
import logging
from pathlib import Path

from flask import Flask, send_from_directory

import config

WEB_DIR = Path(__file__).resolve().parent.parent / "frontend" / "web"


def create_app() -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(
        __name__, static_folder=str(WEB_DIR), static_url_path=""
    )
    app.config["JSON_SORT_KEYS"] = False

    from api.routes import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    return app
