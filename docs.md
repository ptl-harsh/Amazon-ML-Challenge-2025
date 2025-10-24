Approach & Results

---

# Smart Product Pricing — ML Approach, Experiments, and Findings

## Problem & Metric

* **Task:** Predict product **price** from product **catalog text** (and optional image links).
* **Primary metric:** **SMAPE** (Symmetric Mean Absolute Percentage Error).
* **Target transform:** Models trained mostly on **log-price** (`y_log = log1p(price)`) for stability; predictions are mapped back with `expm1` for metric computation and submission.

---

## Data & Preprocessing

**Inputs (train/test):**

* `sample_id`, `catalog_content` (rich text blob), `image_link` (URLs), and `price` (train only).

**Text cleaning**

* Normalize whitespace, strip control chars.
* Extract **Item Name** (`title`) from the blob via regex.
* Heuristic **brand** extraction from `title` → `brand_heur`.

**Quantity & pack parsing**

* Extract `(Value, Unit)` pairs when present; standardize to:

  * **Weight (g)**, **Volume (ml)**, and **Count**.
* Enhanced parsing for patterns like **“3 x 200 g”**, **“2×500 ml”**, with totals:

  * `qty_g_tot`, `qty_ml_tot`, `pack_cnt_tot`, and per-unit features `per_unit_g`, `per_unit_ml`.

**Light numeric features**

* `len_chars`, `len_words`, `upper_ratio`, and `value_extracted`.

> **Images** were **not** downloaded in the final pipeline to keep runtime/memory robust on hosted kernels. The winning solution is **text+numeric only**.

---

## Feature Engineering (Text → Dense)

* **TF-IDF** on concatenated `title + catalog_content + brand_heur`:

  * Word-level: `ngram=(1,2)`, `min_df≈4–5`, `max_df≈0.95`, `sublinear_tf=True`.
  * Char-level: `ngram=(3,6)` (captures units, brand tokens, SKU patterns).
* **Dimensionality reduction:** **TruncatedSVD** to a dense semantic space (**SVD rank 448**, adjustable 256–512 by RAM).
  The resulting dense text vectors are concatenated with numeric features.

**Target encodings (CV-safe)**

* Stratified by **log-price deciles** (**StratifiedKFold**, 5 folds).
* Smoothed **mean target encodings** (in log space) for:

  * `brand_heur`, `tok1`, `tok2`, `unit_extracted`, and interactions `brand_heur×unit_extracted`, `tok1×unit_extracted`.
* Smoothing parameter **α≈12–15** to dampen long-tail leakage.

---

## Modeling

We trained compact-feature models (dense text SVD + numeric + encodings):

**Linear (fast baselines) — log target**

* **Ridge** (`solver=lsqr`)
* **ElasticNet** (mild L1 for stability)
* **HuberRegressor** (robust to label noise)

**Tree ensembles — log target**

* **LightGBM-GBDT** (the eventual winner).
  Tuned for moderate depth/leafs with early stopping.

**Tree ensembles — (attempt)**

* **LightGBM-DART** (dropout boosting).
  → On this dataset/config it **collapsed** (SMAPE ≈ 195% in CV), so **disabled**.

**Tree ensembles — log target**

* **XGBoost** (hist/gpu_hist), conservative regularization, early stopping.

> (An additional **LGB trained in price space** and CatBoost variant were considered; the results provided below reflect the models we actually ran.)

---

## Cross-Validation Strategy

* **StratifiedKFold (5 folds)** using **log-price deciles** to stabilize fold difficulty.
* All target encodings and OOF predictions are fold-consistent to avoid leakage.
* Early stopping on validation fold for tree models.

---

## Experiments & Results (OOF SMAPE)

*(Lower is better; numbers below are from your final runs.)*

| Model             | Target | OOF SMAPE           |
| ----------------- | ------ | ------------------- |
| Ridge             | log    | **70.24**           |
| ElasticNet        | log    | **58.21**           |
| Huber             | log    | **66.83**           |
| **LightGBM-GBDT** | log    | **48.96** ✅         |
| LightGBM-DART     | log    | **195.85** (failed) |
| XGBoost           | log    | **49.94**           |

**Ensembling (meta-stacker):**

* Simplex-constrained, non-negative weights learned on OOF predictions (NNLS-style init + local SMAPE refinement).
* **Final learned weights:** `[Ridge, ENet, Huber, LGB-GBDT, LGB-DART, XGB] = [0, 0, 0, **1.0**, 0, 0]`
* **Blended OOF SMAPE:** **48.96**, i.e., the stacker selected **LightGBM-GBDT only**, which outperformed all others in CV.

---

## Post-processing & Bias Corrections

All corrections applied in **log space** (then mapped to price):

1. **Residual bias smoothing (additive in log)**
   Compute OOF residuals `res = y_log − p_log`, aggregate and smooth:

   * by **brand** (`brand_heur`)
   * by **unit** (`unit_extracted`)
   * by **brand×unit**
     Add small, conservative bias terms to test predictions (α≈12–14).
     *Effect is small but can reduce systematic over/under-pricing for frequent groups.*

2. **Value range control** (price space)

   * Clip to **[0.01, 5 × (Q3 + 3×IQR)]** to prevent explosions.

---

## Why this Works

* **Text is king** here. Strong TF-IDF + higher-rank **SVD** preserves semantics in a compact dense form that gradient-boosted trees can exploit.
* **Quantity & pack features** reduce label noise by normalizing size/pack multiplicity (critical for pricing).
* **Smoothed target encodings** inject powerful priors for brands/units without leaking folds.
* **LightGBM-GBDT** balances bias/variance very well on these dense features; **XGBoost** is competitive but slightly behind.
  **DART** hurt due to aggressive dropout on this distribution.

---

## Ablation Snapshot (qualitative)

* * Enhanced quantity parsing → noticeable drop in OOF.
* * Target encodings & interactions → consistent gain.
* * SVD rank ↑ (to ~448) → steady gain until memory becomes a concern.
* * LightGBM-GBDT tuning (leaves/learning rate/ES) → best single model (**48.96**).
* * DART → unstable here; removed.
* Meta-stacker → confirmed LGB-GBDT dominance; no blend needed in the final run.

---

## Practical Considerations

* **Kernel safety:** No network calls; no image downloads; dense features capped by **SVD rank** and early stopping.
  If memory is tight, use `N_COMP=256–384` and keep only LGB-GBDT.
* **Repro:** Fixed `SEED`, **StratifiedKFold(5)**, target encodings CV-safe, LightGBM with early stopping.

---

## Conclusions

* The **best CV score** achieved: **SMAPE ≈ 48.96** with **LightGBM-GBDT on (SVD text + engineered numeric + CV target encodings)**.
* Linear models provided useful sanity checks but trailed significantly; **XGBoost** was close second; **DART** underperformed and was removed.
* Residual bias corrections (brand/unit/brand×unit) are a **small stabilizer**, not the main driver.
* Further gains (if desired) likely come from:

  1. pushing **SVD rank** if memory allows (≤512),
  2. modest **LGB hyper-sweeps** around `num_leaves` (176–224) and `learning_rate` (0.028–0.04),
  3. higher-fidelity **quantity parsing** for edge unit cases,
  4. selective **image features** (if infra allows) from robust CLIP embeddings.

---


