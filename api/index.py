"""Vercel entrypoint — exposes the FastAPI app from backend/api/main.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.main import app  # noqa: E402,F401
