from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_DIST_DIR = Path(__file__).parent.parent / "frontend" / "dist"


def create_app(payload: dict, endpoint: str = "/api/tree") -> FastAPI:
    app = FastAPI()

    @app.get(endpoint)
    def get_payload():
        return payload

    @app.get("/api/model")
    def get_model_manifest():
        return {
            "model_type": payload["model_type"],
            "endpoint": endpoint,
        }

    if _DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="static")

    return app
