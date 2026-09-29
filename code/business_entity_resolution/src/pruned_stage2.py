"""
Team Jishnu - v7: learned candidate pruning + stage-2 final matcher (no test recomputation).

Pipeline after this step:
  1. key blocking             -> ~48 candidates per S1          (pipeline.py, unchanged)
  2. stage-1 LightGBM (v6)    -> prunes candidates below probability TAU   (learned pruning / meta-blocking)
  3. stage-2 LightGBM         -> final match decisions, run ONLY on the pruned candidates
  4. one-owner rule + threshold

candidate_pairs.tsv = the pruned set (exactly what the stage-2 model runs inference on).
TAU is chosen on validation as the LARGEST value (smallest candidate set) whose 5-fold CV F0.5 is
within 0.001 of the best TAU. Runs in about 30-40 minutes. Output: Jishnu_outputs/v7.

Usage (Colab):  python pruned_stage2.py
"""
import glob
import json
import os
import shutil
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # noqa: E402

P.WORK_DIR = os.environ.get("JISHNU_WORK", "work")
P.DATA_DIR = os.environ.get("JISHNU_DATA", P.DATA_DIR)
OUT = os.environ.get("JISHNU_OUT", os.path.join(P.WORK_DIR, "final"))
TAUS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
TOLERANCE = 0.001
S2_FEATURES = ["p", "p_rank", "p_top1", "p_top2", "p_gap", "p_gap2", "n_ge05", "n_ge03",
               "p_sum", "p_share", "n_cands", "is_s3", "s3_best", "s2_best"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, seed=42, verbose=-1)
N_TREES = 300


def group_features(df):
    """Context features of each candidate within its S1's FULL key-blocked list (stage-1 outputs)."""
    df = df.rename(columns={"prob": "p"}).sort_values(["s1_id", "p"], ascending=[True, False],
                                                      kind="stable").reset_index(drop=True)
    g = df.groupby("s1_id", sort=False)["p"]
    df["p_rank"] = g.cumcount() + 1
    df["p_top1"] = g.transform("max")
    df["p_top2"] = g.transform(lambda x: x.iloc[1] if len(x) > 1 else 0.0)
    df["p_gap"] = df["p_top1"] - df["p"]
    df["p_gap2"] = df["p"] - g.shift(-1).fillna(0.0)
    df["n_ge05"] = g.transform(lambda x: (x >= 0.5).sum())
    df["n_ge03"] = g.transform(lambda x: (x >= 0.3).sum())
    df["p_sum"] = g.transform("sum")
    df["p_share"] = df["p"] / df["p_sum"].clip(lower=1e-9)
    df["n_cands"] = g.transform("count")
    df["is_s3"] = df["cand_id"].str.startswith("S3").astype(np.float32)
    src_best = df.groupby(["s1_id", "is_s3"], sort=False)["p"].transform("max")
    df["s3_best"] = np.where(df["is_s3"] == 1, df["p"] >= src_best, 0).astype(np.float32)
    df["s2_best"] = np.where(df["is_s3"] == 0, df["p"] >= src_best, 0).astype(np.float32)
    return df


def macro(ids, pred, gold):
    return float(np.mean([P.f05(pred.get(e, set()), gold[e]) for e in ids]))


def cv_stage2(v, ids, gold, folds_of):
    """5-fold CV (by entity) of stage-2 on the pruned pairs v; returns (best F0.5, threshold)."""
    folds = v["s1_id"].map(folds_of).values
    p2 = np.zeros(len(v))
    for k in range(5):
        tr, te = folds != k, folds == k
        m = lgb.train(PARAMS, lgb.Dataset(v.loc[tr, S2_FEATURES], v.loc[tr, "y"]), N_TREES)
        p2[te] = m.predict(v.loc[te, S2_FEATURES])
    best = (-1.0, 0.5)
    for thr in np.arange(0.30, 0.91, 0.05):
        sc = macro(ids, P.decode(v["s1_id"], v["cand_id"], p2, thr), gold)
        if sc > best[0]:
            best = (sc, float(thr))
    return best


# ---------------------------------------------------------------- 1. validation entities + v6 probabilities
P.log("loading v6 validation entities")
s1_ids = P.train_sample()["entity_id"].tolist()
gold_all = P.load_gold()
rng = np.random.default_rng(P.SEED)
val_ids = sorted(set(rng.choice(s1_ids, size=len(s1_ids) // 5, replace=False)))
gold = {e: gold_all.get(e, set()) for e in val_ids}
del gold_all
vset = set(val_ids)
parts = []
for f in sorted(glob.glob(P.path("trainfe", "chunk*.parquet"))):
    d = pd.read_parquet(f)
    parts.append(d[d["s1_id"].isin(vset)])
val = pd.concat(parts, ignore_index=True)
del parts
booster = lgb.Booster(model_file=P.path("model", "model.txt"))
val["prob"] = booster.predict(val[P.FEATURES])
val = group_features(val[["s1_id", "cand_id", "prob", "y"]])
total_true = sum(len(g) for g in gold.values())
cfg = json.load(open(P.path("model", "threshold.json")))
v6_score = macro(val_ids, P.decode(val["s1_id"], val["cand_id"], val["p"], cfg["t_top"], cfg["t_rest"]), gold)
P.log(f"{len(val_ids)} validation entities | v6 rule F0.5 {v6_score:.4f} | "
      f"key-blocked candidates per S1: {len(val) / len(val_ids):.1f}")

# ---------------------------------------------------------------- 2. choose TAU
folds_of = {e: i % 5 for i, e in enumerate(rng.permutation(val_ids))}
results = []
for tau in TAUS:
    v = val[val["p"] >= tau].reset_index(drop=True)
    kept_true = int(v["y"].sum())
    sc, thr = cv_stage2(v, val_ids, gold, folds_of)
    results.append((tau, sc, thr, len(v) / len(val_ids), kept_true / total_true))
    P.log(f"  TAU={tau:.2f}: candidates/S1 {len(v) / len(val_ids):5.2f} | true matches kept "
          f"{kept_true / total_true:.3f} | stage-2 CV F0.5 {sc:.4f} (thr {thr:.2f})")
best_sc = max(r[1] for r in results)
tau, s2_score, s2_thr, val_cands, val_recall = max(r for r in results if r[1] >= best_sc - TOLERANCE)
P.log(f"CHOSEN: TAU={tau:.2f} -> {val_cands:.2f} candidates/S1 (was {len(val) / len(val_ids):.1f}), "
      f"stage-2 CV F0.5 {s2_score:.4f} vs v6 rule {v6_score:.4f}")

# ---------------------------------------------------------------- 3. final stage-2, apply to test
os.makedirs(OUT, exist_ok=True)
v = val[val["p"] >= tau]
final = lgb.train(PARAMS, lgb.Dataset(v[S2_FEATURES], v["y"]), N_TREES)
final.save_model(os.path.join(OUT, "stage2_model.txt"))
test_ids = P.load_prep("test", (1,), ["entity_id"])["entity_id"].tolist()
files = sorted(glob.glob(P.path("testpred", "chunk*.parquet")))
assert len(files) == (len(test_ids) + P.S1_CHUNK - 1) // P.S1_CHUNK, "some v6 test chunks are missing"
c_path = os.path.join(OUT, "candidate_pairs.tsv")
scored, n_cand_pairs = [], 0
with open(c_path + ".tmp", "w", encoding="utf-8") as fc:
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    for i, f in enumerate(files):                      # chunk i = test S1 rows i*S1_CHUNK ...
        d = group_features(pd.read_parquet(f))
        d = d[d["p"] >= tau]                           # learned pruning
        d["p2"] = final.predict(d[S2_FEATURES]) if len(d) else []
        n_cand_pairs += len(d)
        cands = {}
        for a, b in zip(d["s1_id"].values, d["cand_id"].values):
            cands.setdefault(a, []).append(b)
        for e in test_ids[i * P.S1_CHUNK:(i + 1) * P.S1_CHUNK]:
            fc.write(e + "\t" + ",".join(sorted(cands.get(e, ()))) + "\n")
        scored.append(d.loc[d["p2"] >= s2_thr, ["s1_id", "cand_id", "p2"]])
        P.log(f"test chunk {i + 1}/{len(files)} done")
os.replace(c_path + ".tmp", c_path)

scored = pd.concat(scored, ignore_index=True).sort_values("p2", ascending=False, kind="stable")
n_before = len(scored)
scored = scored[~scored["cand_id"].duplicated()]      # one-owner rule on stage-2 probabilities
matches = P.decode(scored["s1_id"], scored["cand_id"], scored["p2"], s2_thr)
P.write_tsv(os.path.join(OUT, "matching_results.tsv"), test_ids, matches, "matched_entity_ids")

stats = {"tau": tau, "stage2_threshold": s2_thr, "stage2_cv_f05": s2_score, "v6_val_f05": v6_score,
         "val_candidates_per_s1_before": len(val) / len(val_ids), "val_candidates_per_s1_after": val_cands,
         "val_true_matches_kept_after_pruning": val_recall,
         "test_candidates_per_s1": n_cand_pairs / len(test_ids),
         "one_owner_removed": n_before - len(scored),
         "tau_table": [{"tau": r[0], "cv_f05": r[1], "cands_per_s1": r[3], "kept": r[4]} for r in results]}
with open(os.path.join(OUT, "v7_stats.json"), "w") as fh:
    json.dump(stats, fh, indent=1)
shutil.copy(os.path.abspath(__file__), os.path.join(OUT, "pruned_stage2.py"))
n_empty = sum(1 for e in test_ids if e not in matches)
P.log(f"TEST: {n_cand_pairs / len(test_ids):.2f} candidates per S1 (v6: ~48) | "
      f"{n_empty / len(test_ids):.1%} predicted as no-match")
P.log(f"v7 written to {OUT}: matching_results.tsv, candidate_pairs.tsv, v7_stats.json")
