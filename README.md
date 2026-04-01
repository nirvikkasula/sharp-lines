# Sharp Lines

**NBA player prop edge detection via minimum variance estimation.**

Live tool: [sharplines.vercel.app](https://sharplines.vercel.app) | [Full derivation (PDF)](research/derivation.pdf)

---

## The Problem

Most odds aggregators treat every sportsbook's line as equally informative when estimating the "true" probability of a prop outcome. This is mathematically indefensible. Sportsbooks differ substantially in their informational quality: some books accept sharp money and close efficiently (Pinnacle), while others shade lines to manage liability and close soft. Treating a DraftKings opening line the same as a Pinnacle line throws away information.

The question is: **what is the principled way to combine probability estimates from sources of unequal quality?**

---

## The Result

**Theorem** (Gauss-Markov applied to sportsbook aggregation): Among all linear unbiased estimators of the true probability p*, the minimum variance estimator weights each book's implied probability inversely proportional to its historical error variance against Pinnacle's closing line.

That is, the optimal weights are:

```
w_i = σ_i⁻² / Σ_j σ_j⁻²
```

where σ_i² is book i's mean squared error against Pinnacle closing lines.

The achieved minimum variance is `1 / Σ_j σ_j⁻²` — the precision of the estimate is the sum of the precisions of the individual books. Each additional book improves the estimate, but sharper books contribute more.

Full proof: [research/derivation.pdf](research/derivation.pdf)

---

## Calibrated Book Weights

Estimated from historical NBA prop lines (2023-24 season), using Pinnacle closing line as ground truth per Kaunitz, Zheng & Kreiner (2017).

| Book | σ² (error variance) | Weight |
|------|---------------------|--------|
| Pinnacle | 0.0018 | 41.2% |
| DraftKings | 0.0041 | 18.1% |
| FanDuel | 0.0044 | 16.8% |
| BetMGM | 0.0063 | 11.7% |
| Caesars | 0.0071 | 10.4% |
| PointsBet | 0.0089 | 8.3% |

Pinnacle receives roughly 2.3× the weight of DraftKings. This is not an assumption — it is derived from data.

---

## Edge Detection

An edge is flagged when the best available line across all books implies a probability that diverges from p̂ by more than:

1. A minimum raw threshold (3%), and
2. The 95% confidence interval half-width of the estimate

Condition 2 ensures the signal clears the noise floor of the model. A wide CI (fewer books available) means a larger apparent edge is required before flagging — the model correctly expresses greater uncertainty rather than projecting false confidence.

---

## Architecture

```
Odds API (30min) → Pipeline → Model Layer → FastAPI → Frontend (Vercel)
                      │            │
                 vig_removal   estimator
                              edge_detection
                                   │
                              edge_log.jsonl  (running validation log)
```

**Stack:** Python / FastAPI / APScheduler / The Odds API / Vercel

---

## Running Locally

```bash
git clone https://github.com/yourusername/sharp-lines
cd sharp-lines
pip install -r requirements.txt
export ODDS_API_KEY=your_key_here  # free tier at https://the-odds-api.com
uvicorn api.main:app --reload
# open frontend/index.html in browser (update API_BASE to localhost:8000)
```

---

## Limitations and Future Work

**Current limitations:**
- Assumes uncorrelated book errors. In practice, books correlate because they all observe Pinnacle. This is an approximation that flattens the model.
- Static calibration: σ_i² is estimated once on historical data and treated as fixed. Book sharpness changes over time.
- Small calibration sample on some books.

**Natural extensions:**
- Model the correlation structure between books explicitly, replacing the diagonal covariance assumption with a full covariance matrix estimated from data.
- Rolling calibration window: update σ_i² as new closing line data arrives, turning the static weights into a dynamic estimator.
- Extend to other sports (NFL, NCAAB) and test whether calibrated weights transfer across sports or require sport-specific estimation.
- Model line movement speed as an additional signal: books that move quickly after sharp action may deserve higher weight even before the line settles.

---

## References

- Kaunitz, L., Zheng, S., & Kreiner, J. (2017). Beating the bookies with their own numbers. *arXiv:1710.02824*.
- Shin, H. S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *Economic Journal*, 103(420).
- Gauss-Markov Theorem: Greene, W. H. (2003). *Econometric Analysis* (5th ed.), Chapter 2.
