"""Flask application factory."""
import logging

from flask import Flask

import config


def create_app() -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    from api.routes import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return {"service": "eth-predictor-api", "status": "ok"}

    return app
