"""Flask API entrypoint: python run_api.py"""
import config
from api import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
