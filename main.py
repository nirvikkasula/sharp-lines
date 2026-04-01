"""
Main Pipeline
=============

Fetches live NBA prop lines, runs them through the minimum variance
estimator, detects edges, and returns structured results for the API.

Can be run directly (python -m pipeline.main) or called from the
FastAPI scheduler.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pipeline.odds_fetcher import fetch_all_props, TRACKED_BOOKS
from pipeline.vig_removal import remove_vig_proportional, implied_to_american
from model.estimator import compute_true_prob, BOOK_WEIGHTS, BOOK_SIGMAS_SQUARED
from model.edge_detection import detect_edge, EdgeResult


@dataclass
class PropEdge:
    player: str
    prop_type: str
    line_value: float
    game: str
    side: str
    best_book: str
    best_american_odds: int
    true_prob: float
    book_implied_prob: float
    edge_magnitude: float
    confidence: float
    ci_lower: float
    ci_upper: float
    books_used: list[str]
    weights_applied: dict[str, float]
    flagged_at: str


def parse_market_name(market: str) -> str:
    return market.replace("player_", "").replace("_", " ").title()


def run_pipeline(raw_events: list[dict] | None = None) -> list[PropEdge]:
    """
    Full pipeline: fetch -> parse -> estimate -> detect -> return edges.

    Args:
        raw_events: optional pre-fetched events (for testing without API key)

    Returns:
        List of PropEdge objects where has_edge=True, sorted by edge magnitude
    """
    if raw_events is None:
        raw_events = fetch_all_props()

    edges = []
    now = datetime.now(timezone.utc).isoformat()

    for event in raw_events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        game_label = f"{away} @ {home}"

        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker["key"]
            if book_key not in TRACKED_BOOKS:
                continue

            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                prop_type = parse_market_name(market_key)

                # Group outcomes by player + line
                player_lines: dict[str, dict] = {}

                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", outcome.get("name", "Unknown"))
                    name = outcome.get("name", "")  # "Over" or "Under"
                    point = outcome.get("point", 0.0)
                    price = outcome.get("price", -110)

                    key = f"{player}|{point}"
                    if key not in player_lines:
                        player_lines[key] = {
                            "player": player,
                            "line": point,
                            "over": {},
                            "under": {},
                        }

                    side = "over" if "over" in name.lower() else "under"
                    player_lines[key][side][book_key] = price

                for key, pl in player_lines.items():
                    if not pl["over"] or not pl["under"]:
                        continue

                    # Build book_probs dict: {book: vig_removed_over_prob}
                    book_probs = {}
                    for book in TRACKED_BOOKS:
                        if book in pl["over"] and book in pl["under"]:
                            p_over, _ = remove_vig_proportional(
                                pl["over"][book], pl["under"][book]
                            )
                            book_probs[book] = p_over

                    if len(book_probs) < 2:
                        continue

                    try:
                        estimate = compute_true_prob(book_probs)
                    except ValueError:
                        continue

                    # Best available: book with highest over prob (most favorable odds)
                    best_over_book = max(book_probs, key=lambda b: book_probs[b])
                    best_over_prob = book_probs[best_over_book]
                    best_over_odds = pl["over"].get(best_over_book, -110)

                    # Best under: book with lowest over prob
                    best_under_book = min(book_probs, key=lambda b: book_probs[b])
                    best_under_prob = 1.0 - book_probs[best_under_book]
                    best_under_odds = pl["under"].get(best_under_book, -110)

                    result = detect_edge(
                        estimate=estimate,
                        best_over_prob=best_over_prob,
                        best_under_prob=best_under_prob,
                        best_over_book=best_over_book,
                        best_under_book=best_under_book,
                    )

                    if result.has_edge:
                        best_odds = best_over_odds if result.side == "over" else best_under_odds
                        edges.append(PropEdge(
                            player=pl["player"],
                            prop_type=prop_type,
                            line_value=pl["line"],
                            game=game_label,
                            side=result.side,
                            best_book=result.best_book,
                            best_american_odds=best_odds,
                            true_prob=result.true_prob,
                            book_implied_prob=result.best_prob,
                            edge_magnitude=result.edge_magnitude,
                            confidence=result.confidence,
                            ci_lower=estimate.confidence_interval[0],
                            ci_upper=estimate.confidence_interval[1],
                            books_used=estimate.books_used,
                            weights_applied=estimate.weights_applied,
                            flagged_at=now,
                        ))

    edges.sort(key=lambda e: e.edge_magnitude, reverse=True)
    return edges


def edges_to_dict(edges: list[PropEdge]) -> list[dict]:
    return [asdict(e) for e in edges]


if __name__ == "__main__":
    import json
    print("Running pipeline...")
    results = run_pipeline()
    print(f"Found {len(results)} edges")
    for edge in results[:5]:
        print(f"  {edge.player} {edge.prop_type} {edge.line_value} "
              f"{edge.side.upper()} @ {edge.best_book} "
              f"(edge: {edge.edge_magnitude:.3f}, conf: {edge.confidence:.2f}x CI)")
