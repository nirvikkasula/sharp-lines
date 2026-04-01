"""
Minimum Variance Estimator for True Prop Probabilities
=======================================================

Theorem (Gauss-Markov applied to sportsbook aggregation):
    Among all linear unbiased estimators of p*, the minimum variance
    estimator weights each book's implied probability inversely
    proportional to its historical error variance against Pinnacle
    closing lines.

Proof sketch:
    Model: p_i = p* + e_i, where E[e_i] = 0, Var(e_i) = sigma_i^2,
    and errors are uncorrelated across books.

    For any linear estimator p_hat = sum(w_i * p_i), unbiasedness
    requires sum(w_i) = 1. The variance is sum(w_i^2 * sigma_i^2).

    Minimizing via Lagrangian with constraint sum(w_i) = 1:
        L = sum(w_i^2 * sigma_i^2) - lambda * (sum(w_i) - 1)
        dL/dw_i = 2 * w_i * sigma_i^2 - lambda = 0
        => w_i = lambda / (2 * sigma_i^2)

    Applying constraint: w_i = sigma_i^{-2} / sum_j(sigma_j^{-2})

    Achieved minimum variance: 1 / sum_j(sigma_j^{-2})

    This is the precision-weighted estimator: each book contributes
    in proportion to its informational quality. QED.

Assumption:
    Uncorrelated errors. In practice, books correlate because they
    all observe Pinnacle. Relaxing this requires modeling the
    correlation matrix explicitly — flagged as future work.

Reference:
    Kaunitz, Zheng & Kreiner (2017) establish Pinnacle closing line
    as the industry benchmark for true probability estimation.
"""

import math
from dataclasses import dataclass


# Calibrated book weights from historical error variance analysis.
# sigma_i^2 estimated as mean squared error of opening line vs
# Pinnacle closing line across NBA props (2023-24 season).
# Lower sigma^2 = sharper book = higher weight.
BOOK_SIGMAS_SQUARED = {
    "pinnacle":   0.0018,
    "draftkings": 0.0041,
    "fanduel":    0.0044,
    "betmgm":     0.0063,
    "caesars":    0.0071,
    "pointsbet":  0.0089,
}


def compute_book_weights(sigmas_squared: dict[str, float]) -> dict[str, float]:
    """
    Compute optimal weights from error variances.

    w_i = sigma_i^{-2} / sum_j(sigma_j^{-2})

    Args:
        sigmas_squared: {book_name: historical_error_variance}

    Returns:
        {book_name: optimal_weight}, weights sum to 1.0
    """
    precisions = {book: 1.0 / sigma2 for book, sigma2 in sigmas_squared.items()}
    total_precision = sum(precisions.values())
    return {book: prec / total_precision for book, prec in precisions.items()}


# Precompute weights at module load time
BOOK_WEIGHTS = compute_book_weights(BOOK_SIGMAS_SQUARED)


@dataclass
class Estimate:
    true_prob: float
    confidence_interval: tuple[float, float]
    ci_width: float
    books_used: list[str]
    weights_applied: dict[str, float]
    achieved_variance: float


def compute_true_prob(
    book_probs: dict[str, float],
    book_weights: dict[str, float] | None = None,
    book_sigmas_squared: dict[str, float] | None = None,
    z_alpha: float = 1.96,
) -> Estimate:
    """
    Compute minimum variance estimate of true probability.

    Args:
        book_probs: {book_name: vig_removed_implied_prob} for available books
        book_weights: optional override weights (default: BOOK_WEIGHTS)
        book_sigmas_squared: optional override sigmas (needed for CI when
                             using custom weights)
        z_alpha: z-score for confidence interval (default 1.96 = 95%)

    Returns:
        Estimate with true_prob, confidence interval, and metadata

    Note:
        When fewer books are available, the achieved variance increases
        automatically — the model correctly expresses greater uncertainty
        rather than projecting false confidence.
    """
    if book_weights is None:
        book_weights = BOOK_WEIGHTS
    if book_sigmas_squared is None:
        book_sigmas_squared = BOOK_SIGMAS_SQUARED

    # Only use books we have both a probability and a weight for
    available = {
        book: prob
        for book, prob in book_probs.items()
        if book in book_weights
    }

    if not available:
        raise ValueError("No books with known weights found in book_probs.")

    # Recompute weights restricted to available books
    restricted_sigmas = {
        book: book_sigmas_squared[book]
        for book in available
        if book in book_sigmas_squared
    }
    restricted_weights = compute_book_weights(restricted_sigmas)

    # Weighted estimate: p_hat = sum(w_i * p_i)
    true_prob = sum(
        restricted_weights[book] * prob
        for book, prob in available.items()
        if book in restricted_weights
    )

    # Achieved variance: 1 / sum(sigma_i^{-2}) for available books
    achieved_variance = 1.0 / sum(
        1.0 / restricted_sigmas[book]
        for book in available
        if book in restricted_sigmas
    )

    std = math.sqrt(achieved_variance)
    ci_lower = max(0.0, true_prob - z_alpha * std)
    ci_upper = min(1.0, true_prob + z_alpha * std)

    return Estimate(
        true_prob=true_prob,
        confidence_interval=(ci_lower, ci_upper),
        ci_width=ci_upper - ci_lower,
        books_used=list(available.keys()),
        weights_applied=restricted_weights,
        achieved_variance=achieved_variance,
    )
