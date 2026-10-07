# Loom script (2:45)

About 380 spoken words, at roughly 140 words per minute. **[Brackets]** are what to show on screen.

---

### 0:00–0:30 · Key findings
**[Report page 2: Figure 2, then Figure 3]**

> The task is to price freight loads for November and December using January to October, so it's a forecast one to two months ahead, not a random split.
> Distance explains most of the price. market_index is a daily market signal with a weekly cycle: low on Sunday and Monday, peaking Thursday.
> But even after market_index, prices drift up about six percent over the year, and they ramp through every quarter-end month, then reset on the first. December is a quarter-end month.
> Eight validation cities never appear in training. And quote_signal turned out to be noise, so I dropped it.

### 0:30–1:00 · Data problems and fixes
**[Figure 5: the corrupted-label histogram, then the data-quality table]**

> The biggest problem: 677 labels, about 1.4 percent, are off by a multiplier, roughly 3.3 times or 0.29 times the expected price. They form two clean clusters, so a robust residual fit finds them. I drop them from training only, refitting the detector inside every fold.
> Weights have sign flips, a cap at 47,500, a stuck floor and gaps. I take the absolute value, add flags and impute. Missing market_index gets that day's average.

### 1:00–1:40 · Why this model
**[Figure 7: the backtest error lines, then the model table]**

> The model has two stages. First, a linear model on log price that carries the trend and the quarter-end ramp. Then LightGBM on its residuals, with no trend features.
> Why not just boosted trees? A tree predicts a flat line past the last date it saw. Here, its error slides downward through the September ramp.
> Head to head, the fold-average MAE is $85 for the hybrid against $114 for an XGBoost-LightGBM blend. With eight cities withheld, the hybrid holds at $88 and the tree model falls to $111.

### 1:40–2:10 · Validation approach
**[Figure 6: the fold diagram, then the random-vs-forward table]**

> I used expanding forward folds that test August, September and October, plus a block that trains through August and tests September–October, which stands in for November–December.
> A random split would have fooled me: there the two models tie, $92 against $95. The gap only shows when you predict forward.

### 2:10–2:45 · Code walkthrough
**[The editor: open each file as it's named]**

> `clean.py` holds the input repairs and the corrupted-label detector. `models.py` has HybridModel: stage one, stage two, and the routing for unseen cities. `validate.py` runs the folds and the metrics. The tuning grid was committed before I ran it.
> `run_all.py` runs everything end to end: it trains on all ten months, writes the predictions and the December chart, runs Spotter's scorer, and builds this report.
> **[candidate_december.png]** December rises from $839 to $881, with the weekly cycle on top of the quarter-end ramp.
