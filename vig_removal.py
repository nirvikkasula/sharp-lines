"""
Vig removal utilities.

Converts American odds to vig-removed implied probabilities using
proportional normalization. Shin (1993) method noted as future work.
"""


def american_to_implied(odds: int) -> float:
    """Convert American odds to raw implied probability (before vig removal)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def remove_vig_proportional(over_odds: int, under_odds: int) -> tuple[float, float]:
    """
    Remove vig using proportional normalization.

    For a two-outcome prop, raw implied probabilities sum to > 1.
    We normalize so they sum to 1, preserving their ratio.

    Args:
        over_odds: American odds for the over
        under_odds: American odds for the under

    Returns:
        (p_over, p_under): vig-removed implied probabilities

    Limitation:
        Assumes symmetric vig structure. Shin (1993) provides a more
        sophisticated model that accounts for asymmetric vig as a function
        of outcome probability — flagged as future work.
    """
    q_over = american_to_implied(over_odds)
    q_under = american_to_implied(under_odds)
    total = q_over + q_under

    p_over = q_over / total
    p_under = q_under / total

    return p_over, p_under


def implied_to_american(prob: float) -> int:
    """Convert implied probability back to American odds (for display)."""
    if prob >= 0.5:
        return round(-prob / (1 - prob) * 100)
    else:
        return round((1 - prob) / prob * 100)
