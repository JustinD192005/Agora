"""CORS configuration helper.

Reads AGORA_CORS_ORIGINS as a comma-separated list of origins. If unset,
falls back to localhost dev defaults so frontend devs don't need to
configure anything to spin up Vite/Next/CRA.

Set AGORA_CORS_ORIGINS=* to allow any origin (credentials disabled — the
CORS spec forbids wildcard + credentials).
"""
import os

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = structlog.get_logger()


# Common dev server ports out of the box.
DEFAULT_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]


def configure_cors(app: FastAPI) -> None:
    raw = os.getenv("AGORA_CORS_ORIGINS", "").strip()

    if raw == "*":
        origins = ["*"]
        # Wildcard origin can't coexist with credentials per the CORS spec.
        allow_credentials = False
    elif raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        allow_credentials = True
    else:
        origins = DEFAULT_DEV_ORIGINS
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Run-Id"],
    )

    log.info("cors.configured", origins=origins, allow_credentials=allow_credentials)
