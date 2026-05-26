from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_DIST_DIR = Path(__file__).parent.parent / "frontend" / "dist"


def create_app(payload: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/api/tree")
    def get_tree():
        return payload

    if _DIST_DIR.exists():
        app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="static")

    return app
