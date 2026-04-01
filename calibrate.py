"""
calibrate.py
============

Estimates sigma_squared (mean squared error vs Pinnacle closing line)
for each book from historical prop data, then computes optimal weights.

Usage:
    python calibrate.py historical_props.csv

CSV format (required columns):
    book                  - sportsbook key (e.g. draftkings)
    over_odds             - opening American odds for the over
    under_odds            - opening American odds for the under
    pinnacle_close_over   - Pinnacle closing American odds for the over
    pinnacle_close_under  - Pinnacle closing American odds for the under
"""

import csv
import sys
from collections import defaultdict

from pipeline.vig_removal import remove_vig_proportional


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader, 1):
            try:
                rows.append({
                    "book":                row["book"].strip().lower(),
                    "over_odds":           int(row["over_odds"]),
                    "under_odds":          int(row["under_odds"]),
                    "pinnacle_close_over": int(row["pinnacle_close_over"]),
                    "pinnacle_close_under":int(row["pinnacle_close_under"]),
                })
            except (KeyError, ValueError) as e:
                print(f"  Skipping row {i}: {e}")
        return rows


def compute_sigmas(rows: list[dict]) -> dict[str, float]:
    """
    For each book, compute MSE of its vig-removed opening over-prob
    against Pinnacle's vig-removed closing over-prob.
    """
    errors: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        try:
            book_p_over, _ = remove_vig_proportional(row["over_odds"], row["under_odds"])
            pin_p_over, _  = remove_vig_proportional(row["pinnacle_close_over"], row["pinnacle_close_under"])
        except ZeroDivisionError:
            continue

        errors[row["book"]].append((book_p_over - pin_p_over) ** 2)

    return {book: sum(sq) / len(sq) for book, sq in errors.items() if sq}


def compute_weights(sigmas_squared: dict[str, float]) -> dict[str, float]:
    precisions = {book: 1.0 / s2 for book, s2 in sigmas_squared.items()}
    total = sum(precisions.values())
    return {book: p / total for book, p in precisions.items()}


def main(path: str) -> None:
    print(f"Loading {path}...")
    rows = load_csv(path)
    print(f"Loaded {len(rows)} rows.\n")

    sigmas = compute_sigmas(rows)
    weights = compute_weights(sigmas)

    # Sort by weight descending
    books = sorted(weights, key=lambda b: weights[b], reverse=True)

    col = 14
    print(f"{'Book':<{col}}  {'sigma_squared':>15}  {'weight':>8}  {'n_rows':>6}")
    print("-" * (col + 36))

    book_counts = {b: sum(1 for r in rows if r["book"] == b) for b in books}
    for book in books:
        print(f"{book:<{col}}  {sigmas[book]:>15.6f}  {weights[book]:>7.1%}  {book_counts[book]:>6}")

    print()
    print("# Paste into model/estimator.py -> BOOK_SIGMAS_SQUARED")
    print("BOOK_SIGMAS_SQUARED = {")
    for book in books:
        print(f'    "{book}":{" " * (col - len(book) - 1)}{sigmas[book]:.4f},')
    print("}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py historical_props.csv")
        sys.exit(1)
    main(sys.argv[1])
