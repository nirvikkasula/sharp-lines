"""
Sharp Lines API
===============

FastAPI backend. Exposes a single /edges endpoint that returns current
NBA prop edges detected by the pipeline.

APScheduler refreshes the cache every 30 minutes so the frontend always
gets low-latency responses without hammering The Odds API.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.main import run_pipeline, edges_to_dict
from collect_training_data import collect as collect_training_data

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_cache: dict = {
    "edges": [],
    "last_updated": None,
    "error": None,
}


def _refresh() -> None:
    try:
        edges = run_pipeline()
        _cache["edges"] = edges_to_dict(edges)
        _cache["last_updated"] = datetime.now(timezone.utc).isoformat()
        _cache["error"] = None
        print(f"[scheduler] Refreshed: {len(edges)} edges at {_cache['last_updated']}")
    except Exception as e:
        _cache["error"] = str(e)
        print(f"[scheduler] Refresh failed: {e}")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

scheduler = BackgroundScheduler()


def _collect() -> None:
    try:
        n = collect_training_data()
        print(f"[scheduler] Collected {n} training rows.")
    except Exception as e:
        print(f"[scheduler] Training data collection failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run once on startup, then every 30 minutes
    _refresh()
    _collect()
    scheduler.add_job(_refresh, "interval", minutes=30)
    scheduler.add_job(_collect, "interval", minutes=30)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="Sharp Lines", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/edges")
def get_edges():
    return {
        "edges": _cache["edges"],
        "last_updated": _cache["last_updated"],
        "count": len(_cache["edges"]),
        "error": _cache["error"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "last_updated": _cache["last_updated"]}


@app.post("/refresh")
def force_refresh():
    """Manually trigger a pipeline refresh."""
    _refresh()
    return {"status": "refreshed", "count": len(_cache["edges"]), "error": _cache["error"]}
