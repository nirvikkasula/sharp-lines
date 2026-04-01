"""
FastAPI Backend
===============

Serves the edge detection results and book weight data.
Refreshes props every 30 minutes via APScheduler.

Run locally: uvicorn api.main:app --reload
Deploy: Railway, Render, or Fly.io (all have free tiers)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from pipeline.main import run_pipeline, edges_to_dict
from model.estimator import BOOK_WEIGHTS, BOOK_SIGMAS_SQUARED

# In-memory cache (replace with Postgres in production)
_cache: dict = {
    "edges": [],
    "last_updated": None,
    "edge_log": [],   # running log of all flagged edges for validation
}

LOG_PATH = Path("edge_log.jsonl")


def refresh_edges():
    """Called every 30 minutes by the scheduler."""
    print(f"[{datetime.now(timezone.utc).isoformat()}] Refreshing edges...")
    try:
        edges = run_pipeline()
        edge_dicts = edges_to_dict(edges)
        _cache["edges"] = edge_dicts
        _cache["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Append to running log for backtesting
        with LOG_PATH.open("a") as f:
            for edge in edge_dicts:
                f.write(json.dumps(edge) + "\n")

        print(f"  Found {len(edges)} edges")
    except Exception as e:
        print(f"  Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run once at startup
    refresh_edges()

    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_edges, "interval", minutes=30)
    scheduler.start()

    yield

    scheduler.shutdown()


app = FastAPI(
    title="Sharp Lines",
    description="NBA prop edge detection via minimum variance estimation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/edges")
def get_edges():
    """Return currently flagged edges, sorted by magnitude."""
    return {
        "edges": _cache["edges"],
        "last_updated": _cache["last_updated"],
        "count": len(_cache["edges"]),
    }


@app.get("/weights")
def get_weights():
    """
    Return calibrated book weights and underlying error variances.
    This is the mathematical contribution made visible.
    """
    books = []
    total_precision = sum(1.0 / s for s in BOOK_SIGMAS_SQUARED.values())

    for book, sigma2 in sorted(BOOK_SIGMAS_SQUARED.items(), key=lambda x: x[1]):
        books.append({
            "book": book,
            "sigma_squared": round(sigma2, 6),
            "weight": round(BOOK_WEIGHTS.get(book, 0), 4),
            "precision": round(1.0 / sigma2, 2),
            "relative_sharpness": round((1.0 / sigma2) / total_precision * 100, 1),
        })

    return {
        "books": books,
        "methodology": (
            "Weights are derived via minimum variance estimation (Gauss-Markov). "
            "Each book's weight is proportional to its precision (1/sigma^2), "
            "where sigma^2 is estimated from historical deviation against "
            "Pinnacle closing lines."
        ),
    }


@app.get("/log")
def get_edge_log(limit: int = 100):
    """Return historical edge log for validation analysis."""
    if not LOG_PATH.exists():
        return {"edges": [], "total": 0}

    lines = LOG_PATH.read_text().strip().split("\n")
    lines = [l for l in lines if l]
    recent = lines[-limit:]

    return {
        "edges": [json.loads(l) for l in recent],
        "total": len(lines),
    }


@app.get("/health")
def health():
    return {"status": "ok", "last_updated": _cache["last_updated"]}
