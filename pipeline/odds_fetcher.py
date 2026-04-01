"""
Odds Fetcher
============

Fetches live NBA player prop lines from The Odds API.

Flow:
    1. GET /v4/sports/basketball_nba/events  → list of today's games
    2. For each event, GET /v4/sports/basketball_nba/events/{id}/odds
       with all tracked prop markets and bookmakers

Rate limits: free tier allows ~500 requests/month. We batch all markets
per event in a single request to minimize usage.
"""

import os
import httpx
from typing import Any

BASE_URL = "https://api.the-odds-api.com/v4"

TRACKED_BOOKS = [
    "pinnacle",
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "pointsbet",
]

TRACKED_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
]


def _get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ODDS_API_KEY environment variable is not set. "
            "Get a free key at https://the-odds-api.com"
        )
    return key


def fetch_events(api_key: str) -> list[dict]:
    """Fetch today's NBA events."""
    url = f"{BASE_URL}/sports/basketball_nba/events"
    params = {"apiKey": api_key}
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
    return resp.json()


def fetch_event_props(api_key: str, event_id: str) -> dict[str, Any]:
    """Fetch player prop odds for a single event."""
    url = f"{BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(TRACKED_MARKETS),
        "oddsFormat": "american",
        "bookmakers": ",".join(TRACKED_BOOKS),
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
    return resp.json()


def fetch_all_props() -> list[dict]:
    """
    Fetch prop odds for all live NBA events.

    Returns:
        List of event dicts, each with a 'bookmakers' key containing
        prop markets from all tracked books.
    """
    api_key = _get_api_key()
    events = fetch_events(api_key)

    if not events:
        print("No live NBA events found.")
        return []

    print(f"Found {len(events)} events. Fetching props...")
    enriched = []
    for event in events:
        event_id = event["id"]
        try:
            props = fetch_event_props(api_key, event_id)
            enriched.append(props)
        except httpx.HTTPStatusError as e:
            print(f"  Skipping event {event_id}: HTTP {e.response.status_code}")
        except Exception as e:
            print(f"  Skipping event {event_id}: {e}")

    print(f"Fetched props for {len(enriched)} events.")
    return enriched
