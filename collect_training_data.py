"""
collect_training_data.py
========================

Snapshots current NBA prop lines from all tracked books and appends
them to historical_props.csv for use by calibrate.py.

Each run records a UTC timestamp so opening vs. closing can be
distinguished by timestamp order when the same game appears in
multiple snapshots.

Usage:
    python collect_training_data.py                  # appends to historical_props.csv
    python collect_training_data.py --out my.csv     # custom output path
"""

import csv
import os
import sys
from datetime import datetime, timezone

from pipeline.odds_fetcher import fetch_all_props, TRACKED_BOOKS

CSV_PATH = os.path.join(os.path.dirname(__file__), "historical_props.csv")
COLUMNS = ["timestamp", "player", "prop_type", "line_value", "book", "over_odds", "under_odds"]


def parse_market_name(market_key: str) -> str:
    return market_key.replace("player_", "").replace("_", " ").title()


def collect(out_path: str = CSV_PATH) -> int:
    """
    Fetch current lines and append to out_path.

    Returns:
        Number of rows written.
    """
    raw_events = fetch_all_props()
    if not raw_events:
        print("[collector] No events — nothing written.")
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []

    for event in raw_events:
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker["key"]
            if book_key not in TRACKED_BOOKS:
                continue

            for market in bookmaker.get("markets", []):
                prop_type = parse_market_name(market["key"])

                # Group outcomes by player + line_value to pair over/under
                player_lines: dict[str, dict] = {}
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", outcome.get("name", "Unknown"))
                    point = outcome.get("point", 0.0)
                    price = outcome.get("price", -110)
                    name = outcome.get("name", "").lower()

                    key = f"{player}|{point}"
                    if key not in player_lines:
                        player_lines[key] = {"player": player, "line": point, "over": None, "under": None}

                    if "over" in name:
                        player_lines[key]["over"] = price
                    elif "under" in name:
                        player_lines[key]["under"] = price

                for pl in player_lines.values():
                    if pl["over"] is None or pl["under"] is None:
                        continue
                    rows.append({
                        "timestamp":  timestamp,
                        "player":     pl["player"],
                        "prop_type":  prop_type,
                        "line_value": pl["line"],
                        "book":       book_key,
                        "over_odds":  pl["over"],
                        "under_odds": pl["under"],
                    })

    write_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[collector] Wrote {len(rows)} rows to {out_path} (timestamp={timestamp})")
    return len(rows)


if __name__ == "__main__":
    path = CSV_PATH
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        path = sys.argv[idx + 1]
    collect(out_path=path)
