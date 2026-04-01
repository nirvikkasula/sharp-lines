"""
Edge Detection
==============

An edge exists when the best available line across all books implies
a probability that meaningfully diverges from our true probability estimate,
accounting for estimation uncertainty.

We only flag edges where the divergence exceeds the confidence interval
half-width — i.e., where the signal clears the noise floor of our model.
"""

from dataclasses import dataclass
from model.estimator import Estimate


@dataclass
class EdgeResult:
    has_edge: bool
    side: str | None           # "over" or "under"
    best_book: str | None
    best_prob: float | None    # best available implied prob (vig-removed)
    true_prob: float
    edge_magnitude: float      # true_prob - best_prob (positive = edge)
    confidence: float          # how many CI half-widths the edge exceeds
    estimate: Estimate


def detect_edge(
    estimate: Estimate,
    best_over_prob: float,
    best_under_prob: float,
    best_over_book: str,
    best_under_book: str,
    min_edge: float = 0.03,
    require_clears_ci: bool = True,
) -> EdgeResult:
    """
    Detect whether an edge exists on either side of a prop.

    Args:
        estimate: Estimate object from compute_true_prob
        best_over_prob: best vig-removed over probability available
        best_under_prob: best vig-removed under probability available
        best_over_book: book offering best over line
        best_under_book: book offering best under line
        min_edge: minimum raw edge magnitude to flag (default 3%)
        require_clears_ci: if True, edge must exceed CI half-width

    Returns:
        EdgeResult with edge details

    Logic:
        Over edge: true_prob > best_over_prob (book undervaluing over)
        Under edge: (1 - true_prob) > best_under_prob (book undervaluing under)

        We only flag when the edge exceeds both min_edge and (if
        require_clears_ci) the uncertainty in our estimate.
    """
    ci_half = estimate.ci_width / 2.0

    over_edge = estimate.true_prob - best_over_prob
    under_edge = (1.0 - estimate.true_prob) - best_under_prob

    def _is_real(edge_mag: float) -> bool:
        if edge_mag < min_edge:
            return False
        if require_clears_ci and edge_mag < ci_half:
            return False
        return True

    over_real = _is_real(over_edge)
    under_real = _is_real(under_edge)

    # Take the larger edge if both exist
    if over_real and under_real:
        if over_edge >= under_edge:
            under_real = False
        else:
            over_real = False

    if over_real:
        confidence = over_edge / ci_half if ci_half > 0 else float("inf")
        return EdgeResult(
            has_edge=True,
            side="over",
            best_book=best_over_book,
            best_prob=best_over_prob,
            true_prob=estimate.true_prob,
            edge_magnitude=over_edge,
            confidence=confidence,
            estimate=estimate,
        )
    elif under_real:
        confidence = under_edge / ci_half if ci_half > 0 else float("inf")
        return EdgeResult(
            has_edge=True,
            side="under",
            best_book=best_under_book,
            best_prob=best_under_prob,
            true_prob=1.0 - estimate.true_prob,
            edge_magnitude=under_edge,
            confidence=confidence,
            estimate=estimate,
        )
    else:
        return EdgeResult(
            has_edge=False,
            side=None,
            best_book=None,
            best_prob=None,
            true_prob=estimate.true_prob,
            edge_magnitude=max(over_edge, under_edge),
            confidence=0.0,
            estimate=estimate,
        )
