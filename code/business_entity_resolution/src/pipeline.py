"""
Team Jishnu - Amazon ML Challenge 2026 - Business Entity Resolution  (v6)
v6 changes vs v5 (driven by the v5 deep dive: 60% of missed matches were cut by the TOP_K ranking):
  - candidates ranked by RARITY-weighted shared keys (a shared rare building/number counts far more
    than a shared common word), empty names no longer push a candidate down
  - TOP_K 50 / PRE_K 150, TRAIN_SAMPLE 250k
  - new features: key_idf, idf_rank, idf_gap, address/name containment
  - reuses the v5 blocking index and the v3 cleaned data (no rebuild)

v5 changes vs v3 (driven by the diag.py deep dive):
  - address-based blocking keys (rare locality words, number pairs, number+street,
    consecutive address words): India recall was 0.68 vs US 0.89
  - concatenated-name key, digit->letter fixes (r0yce, universa1)
  - better candidate pre-ranking (name AND address similarity) and TOP_K 40:
    48% of missed matches were found but cut by the TOP_K ranking
  - new features: no-space name ratio, acronym match, empty-name flags, shared rare address words
  - one-owner rule built into the final write (each S2/S3 record -> at most one S1)
  - reuses the cleaned data (prep) of v3, so the prep stage is skipped
v3 changes vs v1: French abbreviations, phonetic + token-pair + address blocking keys,
phonetic similarity features, two-threshold decision rule (top candidate / others).
Scalable, RESUMABLE pipeline (built for millions of records on free Colab).

Every stage saves its results to WORK_DIR (on Google Drive). If Colab disconnects,
just run the same command again: finished stages and finished chunks are skipped.

Stages (run all of them with --stage all):
  1. prep     clean/normalize every source file        -> WORK_DIR/prep/*.parquet
  2. index    blocking-key index of S2+S3 (train, test) -> WORK_DIR/index/*.npy
  3. trainfe  candidates + features for a SAMPLE of train S1, chunk by chunk
  4. train    LightGBM, checkpoint saved every TREES_PER_STEP trees; picks threshold
  5. testfe   candidates + features + predictions for ALL test S1, chunk by chunk
  6. write    output/matching_results.tsv and output/candidate_pairs.tsv

Usage (Colab):
  python pipeline.py --stage all
"""
import argparse
import csv
import glob
import json
import os
import re
import time
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# ------------------------------------------------------------------ config
DATA_DIR = "dataset"                                  # override with --data
WORK_DIR = "work"                                     # override with --work
PREP_ROOT = ""   # optional: reuse cleaned data from another work folder (--prep-root)
INDEX_ROOT = ""  # optional: reuse a blocking index from another work folder (--index-root)

READ_CHUNK = 1_000_000   # rows read at a time from each .tsv
S1_CHUNK = 50_000        # S1 records processed (and saved) per chunk
TRAIN_SAMPLE = 250_000   # train S1 entities used for learning (plenty for LightGBM)
MAX_BLOCK = 200          # blocking keys shared by more records than this are ignored
TOP_K = 50               # candidates kept per S1 record
PRE_K = 150              # candidates pre-scored (by name+address similarity) before the TOP_K cut
TREES_PER_STEP = 100     # save a model checkpoint after every this many trees
MAX_TREES = 600
SEED = 42

TEXT_COLS = ["entity_id", "name_n", "core", "addr_n", "country_n"]
DIG = re.compile(r"\d+")
ADDR_STOP = {"road", "street", "avenue", "lane", "nagar", "colony", "floor", "house", "plot", "near",
             "opposite", "sector", "block", "building", "cross", "main", "north", "south", "east",
             "west", "district", "village", "county", "city", "township", "drive", "suite", "ground",
             "first", "second", "phase", "stage", "extension", "layout", "complex", "tower"}

# ------------------------------------------------------------------ helpers
def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def path(*parts):
    root = WORK_DIR
    if parts and parts[0] == "prep" and PREP_ROOT:
        root = PREP_ROOT
    elif parts and parts[0] == "index" and INDEX_ROOT:
        root = INDEX_ROOT
    p = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def save_parquet(df, p):
    """Write to a temp file then rename, so a half-written file never looks finished."""
    tmp = p + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def save_npy(arr, p):
    tmp = p + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, p)


def mark_done(p):
    with open(p, "w") as f:
        f.write("ok")


# ------------------------------------------------------------------ normalization
ABBR = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "co": "company",
    "inc": "incorporated", "intl": "international", "mfg": "manufacturing",
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "nr": "near", "opp": "opposite",
    "apt": "apartment", "bldg": "building", "fl": "floor", "ste": "suite",
    # French address forms (test set contains France)
    "av": "avenue", "bd": "boulevard", "r": "rue", "pl": "place", "fg": "faubourg",
    "rte": "route", "chem": "chemin", "imp": "impasse", "sq": "square",
    "ets": "etablissements", "cie": "compagnie",
}
LEGAL = {
    "private", "limited", "corporation", "company", "incorporated", "llc", "llp",
    "plc", "the", "sarl", "sas", "sa", "eurl", "sasu", "sci", "snc", "compagnie",
    "etablissements", "gmbh",
}


def normalize(s):
    """Lowercase, strip accents (French!), unify '&', drop punctuation, expand abbreviations."""
    if not isinstance(s, str):
        return ""
    ascii_s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = (ascii_s or s).lower().replace("&", " and ")
    s = re.sub(r"[^\w ]", " ", s)
    return " ".join(ABBR.get(t, t) for t in s.split())


def core_name(name_n):
    core = " ".join(t for t in name_n.split() if t not in LEGAL)
    return core or name_n


# ------------------------------------------------------------------ stage 1: prep
def stage_prep(split):
    for src in (1, 2, 3):
        marker = path("prep", f"{split}_s{src}.DONE")
        if os.path.exists(marker):
            log(f"prep {split} source{src}: already done, skipping")
            continue
        tsv = f"{DATA_DIR}/{split}/{split}_source{src}.tsv"
        reader = pd.read_csv(tsv, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=READ_CHUNK, quoting=csv.QUOTE_NONE)
        for i, df in enumerate(reader):
            out = path("prep", f"{split}_s{src}_part{i:03d}.parquet")
            if os.path.exists(out):
                continue
            df["name_n"] = df["business_name"].map(normalize)
            df["core"] = df["name_n"].map(core_name)
            df["addr_n"] = df["business_address"].map(normalize)
            df["country_n"] = df["country"].str.strip().str.lower()
            save_parquet(df[TEXT_COLS], out)
            log(f"prep {split} source{src} part {i} saved ({len(df)} rows)")
        mark_done(marker)


def load_prep(split, srcs, cols=TEXT_COLS):
    """Load normalized records. Order is fixed (sorted file names) so row numbers are stable."""
    files = sorted(f for s in srcs for f in glob.glob(path("prep", f"{split}_s{s}_part*.parquet")))
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


# ------------------------------------------------------------------ stage 2: blocking index
def phon(t):
    """Rough phonetic code, robust to transliteration/typos: lakshmi / laxmi -> 'lxm'."""
    for a, b in (("ksh", "x"), ("ph", "f"), ("th", "t"), ("sh", "s"), ("ck", "k"), ("q", "k"),
                 ("w", "v"), ("z", "s"), ("y", "i"), ("ee", "i"), ("oo", "u")):
        t = t.replace(a, b)
    if re.search(r"[a-z]", t):                      # r0yce -> royce, universa1 -> universal
        t = t.translate(str.maketrans("0135", "oles"))
    t = re.sub(r"(.)\1+", r"\1", t)
    return t[:1] + re.sub(r"[aeiou]", "", t[1:])


def phon_str(s):
    return " ".join(phon(t) for t in s.split())


def num_norm(t):
    """Numeric token without leading zeros: '09' -> '9'; '50b' stays '50b'."""
    return t.lstrip("0") or "0"


def make_keys(core, addr, country):
    """Blocking keys.
    Name   : rare tokens, phonetic tokens, sorted phonetic pairs, prefixes, concatenated name.
    Address: rare locality words, pairs of numbers, number + following word,
             consecutive word pairs (these work even when the names differ completely)."""
    toks = core.split()
    keys = {f"{country}|t|{t}" for t in toks if len(t) >= 3}
    ph = [phon(t) for t in toks]
    keys.update(f"{country}|f|{p}" for p, t in zip(ph, toks) if len(t) >= 3 and len(p) >= 2)
    head = ph[:4]
    for i in range(len(head)):
        for j in range(i + 1, len(head)):
            a, b = sorted((head[i], head[j]))
            keys.add(f"{country}|pp|{a}|{b}")
    if len(toks) >= 2:
        keys.add(f"{country}|p|{toks[0][:4]}|{toks[1][:4]}")
        s = sorted(toks)
        keys.add(f"{country}|s|{s[0][:4]}|{s[1][:4]}")
    elif toks:
        keys.add(f"{country}|p|{toks[0]}")
    cat = "".join(toks)
    if len(cat) >= 5:
        keys.add(f"{country}|cat|{phon(cat[:12])}")
    first3 = toks[0][:3] if toks else ""

    atoks = addr.split()
    nums = []
    for t in atoks:
        if any(ch.isdigit() for ch in t):
            n = num_norm(t)
            if n not in nums:
                nums.append(n)
    for d in nums:
        if len(d) >= 3:
            keys.add(f"{country}|d|{d}|{first3}")
    for i in range(min(len(nums), 5)):
        for j in range(i + 1, min(len(nums), 5)):
            a, b = sorted((nums[i], nums[j]))
            keys.add(f"{country}|nn|{a}|{b}")
    for i in range(len(atoks) - 1):
        a, b = atoks[i], atoks[i + 1]
        if any(ch.isdigit() for ch in a) and b.isalpha() and len(b) >= 3 and b not in ADDR_STOP:
            keys.add(f"{country}|nw|{num_norm(a)}|{phon(b)}")
    words = [w for w in atoks if w.isalpha() and len(w) >= 3]
    for w in words:
        if len(w) >= 5 and w not in ADDR_STOP:
            keys.add(f"{country}|at|{phon(w)}")
    for i in range(min(len(words) - 1, 6)):
        keys.add(f"{country}|ap|{phon(words[i])}|{phon(words[i + 1])}")
    return keys


def key_hashes(df):
    """Return (hash array, row-number array) for all blocking keys of df."""
    hs, rows = [], []
    for start in range(0, len(df), 1_000_000):
        part = df.iloc[start:start + 1_000_000]
        k_list, r_list = [], []
        for r, (c, a, co) in enumerate(zip(part["core"], part["addr_n"], part["country_n"]), start):
            ks = make_keys(c, a, co)
            k_list.extend(ks)
            r_list.extend([r] * len(ks))
        if k_list:
            hs.append(pd.util.hash_array(np.array(k_list, dtype=object)))
            rows.append(np.array(r_list, dtype=np.int32))
    if not hs:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int32)
    return np.concatenate(hs), np.concatenate(rows)


def stage_index(split):
    h_path, i_path = path("index", f"{split}_hash.npy"), path("index", f"{split}_row.npy")
    if os.path.exists(h_path) and os.path.exists(i_path):
        log(f"index {split}: already done, skipping")
        return
    other = load_prep(split, (2, 3), ["core", "addr_n", "country_n"])
    log(f"index {split}: building keys for {len(other)} S2+S3 records")
    h, rows = key_hashes(other)
    del other                                     # free memory before sorting
    log(f"index {split}: {len(h)} raw key entries, sorting")
    order = np.argsort(h, kind="stable")          # memory-friendly: sort once, count runs
    h = h[order]
    rows = rows[order]
    del order
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    counts = np.diff(np.r_[starts, len(h)])
    keep = np.repeat(counts <= MAX_BLOCK, counts)  # drop very common keys (e.g. 'restaurant')
    h, rows = h[keep], rows[keep]
    save_npy(rows, i_path)
    save_npy(h, h_path)
    log(f"index {split}: saved {len(h)} key entries")


# ------------------------------------------------------------------ candidates + features
FEATURES = [
    "shared_keys", "name_ratio", "name_tsort", "name_tset", "name_partial",
    "core_ratio", "core_tset", "name_jacc", "first_tok_eq", "len_diff",
    "addr_ratio", "addr_tset", "addr_partial", "digit_jacc", "digit_any",
    "is_s3", "rank_name", "rank_addr", "name_gap", "addr_gap", "n_cands",
    "phon_tset", "phon_ratio", "phon_first_eq", "core_partial", "phon_gap",
    "nospace_ratio", "acronym_eq", "name_empty", "addr_rare_shared", "num_shared",
    "key_idf", "idf_rank", "idf_gap", "addr_contain", "name_contain",
]


def pair_scores(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def candidates(s1c, other, h_sorted, row_sorted, w_sorted):
    """Look up S1 keys in the S2+S3 index; keep TOP_K candidates per S1 record."""
    q, q_row = key_hashes(s1c)
    lo = np.searchsorted(h_sorted, q, "left")
    hi = np.searchsorted(h_sorted, q, "right")
    cnt = hi - lo
    total = int(cnt.sum())
    if total == 0:
        return None
    s1r = np.repeat(q_row, cnt)
    offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pos = np.repeat(lo, cnt) + offs
    cand = row_sorted[pos]
    wts = w_sorted[pos]                                    # rarity weight of each matched key

    combo = s1r * np.int64(len(other)) + cand
    combo, inv, shared = np.unique(combo, return_inverse=True, return_counts=True)
    idf = np.bincount(inv, weights=wts).astype(np.float32)  # rarity-weighted shared keys
    s1r, cand = combo // len(other), combo % len(other)

    # cheap pre-cut: keep the PRE_K candidates with the highest rarity-weighted overlap per S1
    order = np.lexsort((-idf, s1r))
    s1r, cand, shared, idf = s1r[order], cand[order], shared[order], idf[order]
    first = np.r_[True, s1r[1:] != s1r[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(s1r)), 0))
    keep = (np.arange(len(s1r)) - grp_start) < PRE_K
    s1r, cand, shared, idf = s1r[keep], cand[keep], shared[keep], idf[keep]
    # pre-score with BOTH name and address similarity (v3 used name only)
    tset = pair_scores(s1c["name_n"].values[s1r].tolist(),
                       other["name_n"].values[cand].tolist(), fuzz.token_set_ratio)
    aset = pair_scores(s1c["addr_n"].values[s1r].tolist(),
                       other["addr_n"].values[cand].tolist(), fuzz.token_set_ratio)
    empty = (s1c["core"].values[s1r] == "") | (other["core"].values[cand] == "")
    tset = np.where(empty, 50.0, tset)                     # an empty name is unknown, not a mismatch
    score = idf + (tset + aset) / 100.0
    order = np.lexsort((-score, s1r))
    s1r, cand, shared, idf = s1r[order], cand[order], shared[order], idf[order]
    first = np.r_[True, s1r[1:] != s1r[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(s1r)), 0))
    keep = (np.arange(len(s1r)) - grp_start) < TOP_K
    return s1r[keep], cand[keep], shared[keep], idf[keep]


def chunk_features(s1c, other, h_sorted, row_sorted, w_sorted):
    """All candidate pairs for one S1 chunk, with similarity features."""
    res = candidates(s1c, other, h_sorted, row_sorted, w_sorted)
    if res is None:
        return pd.DataFrame(columns=["s1_id", "cand_id"] + FEATURES)
    s1r, cand, shared, idf = res
    A = {c: s1c[c].values[s1r].tolist() for c in ("name_n", "core", "addr_n")}
    B = {c: other[c].values[cand].tolist() for c in ("name_n", "core", "addr_n")}
    p = pd.DataFrame({"s1_id": s1c["entity_id"].values[s1r],
                      "cand_id": other["entity_id"].values[cand],
                      "shared_keys": shared.astype(np.float32),
                      "key_idf": idf})
    p["name_ratio"] = pair_scores(A["name_n"], B["name_n"], fuzz.ratio)
    p["name_tsort"] = pair_scores(A["name_n"], B["name_n"], fuzz.token_sort_ratio)
    p["name_tset"] = pair_scores(A["name_n"], B["name_n"], fuzz.token_set_ratio)
    p["name_partial"] = pair_scores(A["name_n"], B["name_n"], fuzz.partial_ratio)
    p["core_ratio"] = pair_scores(A["core"], B["core"], fuzz.ratio)
    p["core_tset"] = pair_scores(A["core"], B["core"], fuzz.token_set_ratio)
    p["addr_ratio"] = pair_scores(A["addr_n"], B["addr_n"], fuzz.ratio)
    p["addr_tset"] = pair_scores(A["addr_n"], B["addr_n"], fuzz.token_set_ratio)
    p["addr_partial"] = pair_scores(A["addr_n"], B["addr_n"], fuzz.partial_ratio)

    jac, dj, da, fte, ld = [], [], [], [], []
    for x, y, u, v in zip(A["core"], B["core"], A["addr_n"], B["addr_n"]):
        sx, sy = set(x.split()), set(y.split())
        un = len(sx | sy)
        jac.append(len(sx & sy) / un if un else 0.0)
        du = {num_norm(x) for x in DIG.findall(u)}
        dv = {num_norm(x) for x in DIG.findall(v)}
        dun = len(du | dv)
        dj.append(len(du & dv) / dun if dun else -1.0)   # -1 = no numbers on either side
        da.append(float(bool(du & dv)))
        fte.append(float(x.split()[:1] == y.split()[:1]))
        ld.append(abs(len(x) - len(y)))
    p["name_jacc"], p["digit_jacc"], p["digit_any"] = jac, dj, da
    p["first_tok_eq"], p["len_diff"] = fte, ld

    pa, pb = [phon_str(x) for x in A["core"]], [phon_str(x) for x in B["core"]]
    p["phon_tset"] = pair_scores(pa, pb, fuzz.token_set_ratio)
    p["phon_ratio"] = pair_scores(pa, pb, fuzz.ratio)
    p["phon_first_eq"] = [float(x.split()[:1] == y.split()[:1]) for x, y in zip(pa, pb)]
    p["core_partial"] = pair_scores(A["core"], B["core"], fuzz.partial_ratio)
    p["nospace_ratio"] = pair_scores([x.replace(" ", "") for x in A["core"]],
                                     [x.replace(" ", "") for x in B["core"]], fuzz.ratio)
    acr, emp, ars, nsh = [], [], [], []
    for x, y, u, v in zip(A["core"], B["core"], A["addr_n"], B["addr_n"]):
        ix, iy = "".join(w[0] for w in x.split()), "".join(w[0] for w in y.split())
        acr.append(float((len(ix) >= 2 and ix == y.replace(" ", "")) or
                         (len(iy) >= 2 and iy == x.replace(" ", ""))))
        emp.append(float(not x) + float(not y))
        wu = {w for w in u.split() if len(w) >= 5 and w.isalpha() and w not in ADDR_STOP}
        wv = {w for w in v.split() if len(w) >= 5 and w.isalpha() and w not in ADDR_STOP}
        ars.append(len(wu & wv))
        nsh.append(len({num_norm(t) for t in u.split() if any(c.isdigit() for c in t)} &
                       {num_norm(t) for t in v.split() if any(c.isdigit() for c in t)}))
    p["acronym_eq"], p["name_empty"], p["addr_rare_shared"], p["num_shared"] = acr, emp, ars, nsh

    def contain(x, y):
        """Share of the SHORTER text's words found in the longer one (partial addresses)."""
        sx, sy = set(x.split()), set(y.split())
        m = min(len(sx), len(sy))
        return len(sx & sy) / m if m else -1.0

    p["addr_contain"] = [contain(x, y) for x, y in zip(A["addr_n"], B["addr_n"])]
    p["name_contain"] = [contain(x, y) for x, y in zip(A["core"], B["core"])]
    p["is_s3"] = p["cand_id"].str.startswith("S3").astype(np.float32)
    g = p.groupby("s1_id")
    p["rank_name"] = g["name_tset"].rank(ascending=False, method="min")
    p["rank_addr"] = g["addr_tset"].rank(ascending=False, method="min")
    p["name_gap"] = g["name_tset"].transform("max") - p["name_tset"]
    p["addr_gap"] = g["addr_tset"].transform("max") - p["addr_tset"]
    p["n_cands"] = g["cand_id"].transform("count")
    p["phon_gap"] = g["phon_tset"].transform("max") - p["phon_tset"]
    p["idf_rank"] = g["key_idf"].rank(ascending=False, method="min")
    p["idf_gap"] = g["key_idf"].transform("max") - p["key_idf"]
    p[FEATURES] = p[FEATURES].astype(np.float32)
    return p


def load_index(split):
    """Sorted key hashes, row numbers, and a rarity weight per entry (rare key -> big weight)."""
    h = np.load(path("index", f"{split}_hash.npy"))
    rows = np.load(path("index", f"{split}_row.npy"))
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    counts = np.diff(np.r_[starts, len(h)])
    w = np.repeat(np.log1p(MAX_BLOCK / counts).astype(np.float32), counts)
    return h, rows, w


# ------------------------------------------------------------------ stage 3: train features
def load_gold():
    gt = pd.read_csv(f"{DATA_DIR}/train/train_ground_truth.tsv", sep="\t",
                     dtype=str, keep_default_na=False)
    return {a: set(filter(None, b.split(","))) for a, b in
            zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def train_sample():
    s1 = load_prep("train", (1,))
    rng = np.random.default_rng(SEED)
    pos = np.sort(rng.choice(len(s1), size=min(TRAIN_SAMPLE, len(s1)), replace=False))
    return s1.iloc[pos].reset_index(drop=True)


def stage_trainfe():
    s1 = train_sample()
    chunks = list(range(0, len(s1), S1_CHUNK))
    if all(os.path.exists(path("trainfe", f"chunk{i:03d}.parquet")) for i in range(len(chunks))):
        log("trainfe: already done, skipping")
        return
    other = load_prep("train", (2, 3))
    h_sorted, row_sorted, w_sorted = load_index("train")
    gold = load_gold()
    for i, start in enumerate(chunks):
        out = path("trainfe", f"chunk{i:03d}.parquet")
        if os.path.exists(out):
            log(f"trainfe chunk {i}: already done, skipping")
            continue
        p = chunk_features(s1.iloc[start:start + S1_CHUNK].reset_index(drop=True),
                           other, h_sorted, row_sorted, w_sorted)
        p["y"] = [int(b in gold.get(a, ())) for a, b in zip(p["s1_id"], p["cand_id"])]
        save_parquet(p, out)
        log(f"trainfe chunk {i + 1}/{len(chunks)} saved ({len(p)} pairs)")


# ------------------------------------------------------------------ stage 4: train model
def f05(pred, gold):
    """Per-entity F0.5 as defined by the challenge (singletons included)."""
    if not gold:
        return 1.0 if not pred else 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(pred), tp / len(gold)
    return 1.25 * pr * rc / (0.25 * pr + rc)


def decode(s1_ids, cand_ids, probs, t_top, t_rest=None):
    """Keep each S1's best candidate if prob >= t_top, any other candidate if prob >= t_rest."""
    t_rest = t_top if t_rest is None else t_rest
    df = pd.DataFrame({"a": np.asarray(s1_ids), "b": np.asarray(cand_ids), "p": np.asarray(probs)})
    df = df.sort_values(["a", "p"], ascending=[True, False], kind="stable")
    top = ~df["a"].duplicated().values
    keep = (top & (df["p"].values >= t_top)) | (df["p"].values >= t_rest)
    out = {}
    for a, b in zip(df["a"].values[keep], df["b"].values[keep]):
        out.setdefault(a, set()).add(b)
    return out


def stage_train():
    thr_path = path("model", "threshold.json")
    if os.path.exists(thr_path):
        log("train: already done, skipping")
        return
    s1_ids = train_sample()["entity_id"].tolist()
    gold = load_gold()
    feats = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(path("trainfe", "chunk*.parquet")))],
                      ignore_index=True)
    total_gold = sum(len(gold.get(e, ())) for e in s1_ids)
    log(f"Blocking recall on sample: {feats['y'].sum() / max(total_gold, 1):.3f} | "
        f"avg candidates/S1: {len(feats) / len(s1_ids):.1f}")

    rng = np.random.default_rng(SEED)
    val_ids = set(rng.choice(s1_ids, size=len(s1_ids) // 5, replace=False))
    is_val = feats["s1_id"].isin(val_ids).values
    dtrain = lgb.Dataset(feats.loc[~is_val, FEATURES], feats.loc[~is_val, "y"], free_raw_data=False)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  seed=SEED, verbose=-1, num_threads=0)

    ckpt = path("model", "model.txt")
    booster = lgb.Booster(model_file=ckpt) if os.path.exists(ckpt) else None
    trees = booster.current_iteration() if booster else 0
    if trees:
        log(f"train: resuming from checkpoint with {trees} trees")
    while trees < MAX_TREES:
        booster = lgb.train(params, dtrain, num_boost_round=TREES_PER_STEP,
                            init_model=booster, keep_training_booster=True)
        tmp = ckpt + ".tmp"
        booster.save_model(tmp)
        os.replace(tmp, ckpt)
        trees = booster.current_iteration()
        log(f"train: checkpoint saved at {trees}/{MAX_TREES} trees")

    val = feats[is_val]
    probs = booster.predict(val[FEATURES])
    val_list = sorted(val_ids)
    floor = np.mean([f05(set(), gold.get(e, set())) for e in val_list])
    log(f"Val F0.5 predicting ALL EMPTY: {floor:.4f}  <- floor to beat")
    def val_score(t_top, t_rest):
        pred = decode(val["s1_id"], val["cand_id"], probs, t_top, t_rest)
        return float(np.mean([f05(pred.get(e, set()), gold.get(e, set())) for e in val_list]))

    best, best_top, best_rest = -1.0, 0.5, 0.5
    for thr in np.arange(0.30, 0.96, 0.05):              # single threshold (v1 style), for reference
        sc = val_score(thr, thr)
        log(f"  single thr={thr:.2f}  val F0.5={sc:.4f}")
        if sc > best:
            best, best_top, best_rest = sc, float(thr), float(thr)
    for t_top in np.arange(0.10, 0.66, 0.05):            # two thresholds: top candidate / others
        for t_rest in np.arange(max(0.50, t_top), 0.96, 0.05):
            sc = val_score(t_top, t_rest)
            if sc > best:
                best, best_top, best_rest = sc, float(t_top), float(t_rest)
                log(f"  new best: t_top={t_top:.2f} t_rest={t_rest:.2f} -> val F0.5 {sc:.4f}")
    with open(thr_path, "w") as f:
        json.dump({"t_top": best_top, "t_rest": best_rest, "val_f05": best}, f)
    log(f"BEST: t_top={best_top:.2f} t_rest={best_rest:.2f} -> val F0.5 {best:.4f} (saved)")


# ------------------------------------------------------------------ stage 5: test predictions
def stage_testfe():
    booster = lgb.Booster(model_file=path("model", "model.txt"))
    s1 = load_prep("test", (1,))
    chunks = list(range(0, len(s1), S1_CHUNK))
    todo = [i for i in range(len(chunks)) if not os.path.exists(path("testpred", f"chunk{i:03d}.parquet"))]
    if not todo:
        log("testfe: already done, skipping")
        return
    log("Countries (test): " + str(s1["country_n"].value_counts().to_dict()))
    other = load_prep("test", (2, 3))
    h_sorted, row_sorted, w_sorted = load_index("test")
    for i in todo:
        start = chunks[i]
        p = chunk_features(s1.iloc[start:start + S1_CHUNK].reset_index(drop=True),
                           other, h_sorted, row_sorted, w_sorted)
        p["prob"] = booster.predict(p[FEATURES]) if len(p) else []
        save_parquet(p[["s1_id", "cand_id", "prob"]], path("testpred", f"chunk{i:03d}.parquet"))
        log(f"testfe chunk {i + 1}/{len(chunks)} saved ({len(p)} pairs)")


# ------------------------------------------------------------------ stage 6: write outputs
def write_tsv(p, s1_ids, mapping, col):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for e in s1_ids:
            f.write(e + "\t" + ",".join(sorted(mapping.get(e, ()))) + "\n")
    os.replace(tmp, p)


def stage_write():
    """Stream both output files chunk by chunk (memory-safe), applying the one-owner rule:
    training ground truth links every S2/S3 record to at most ONE S1, so each record goes
    only to the S1 with the highest probability."""
    cfg = json.load(open(path("model", "threshold.json")))
    t_top, t_rest = cfg["t_top"], cfg["t_rest"]
    lo_thr = min(t_top, t_rest)
    s1_ids = load_prep("test", (1,), ["entity_id"])["entity_id"].tolist()
    files = sorted(glob.glob(path("testpred", "chunk*.parquet")))
    assert len(files) == (len(s1_ids) + S1_CHUNK - 1) // S1_CHUNK, "some test chunks are missing"

    best = {}                                          # pass 1: owner of each S2/S3 record
    for f in files:
        d = pd.read_parquet(f)
        d = d[d["prob"] >= lo_thr]
        for a, b, pr in zip(d["s1_id"].values, d["cand_id"].values, d["prob"].values):
            if pr > best.get(b, (None, -1.0))[1]:
                best[b] = (a, pr)

    m_path, c_path = path("output", "matching_results.tsv"), path("output", "candidate_pairs.tsv")
    removed, n_empty = 0, 0
    with open(m_path + ".tmp", "w", encoding="utf-8") as fm, open(c_path + ".tmp", "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, f in enumerate(files):                  # pass 2: chunk i = S1 rows i*S1_CHUNK ...
            d = pd.read_parquet(f)
            cands = {}
            for a, b in zip(d["s1_id"].values, d["cand_id"].values):
                cands.setdefault(a, set()).add(b)
            d = d[d["prob"] >= lo_thr]
            own = np.array([best[b][0] == a for a, b in zip(d["s1_id"].values, d["cand_id"].values)],
                           dtype=bool)
            removed += int((~own).sum())
            matches = decode(d["s1_id"].values[own], d["cand_id"].values[own],
                             d["prob"].values[own], t_top, t_rest)
            for e in s1_ids[i * S1_CHUNK:(i + 1) * S1_CHUNK]:
                m = matches.get(e, ())
                n_empty += not m
                fm.write(e + "\t" + ",".join(sorted(m)) + "\n")
                fc.write(e + "\t" + ",".join(sorted(cands.get(e, ()))) + "\n")
    os.replace(m_path + ".tmp", m_path)
    os.replace(c_path + ".tmp", c_path)
    log(f"write: one-owner rule removed {removed} weaker claims")
    log(f"write: {len(s1_ids)} S1 rows, {n_empty / len(s1_ids):.1%} predicted as no-match, "
        f"t_top {t_top:.2f}, t_rest {t_rest:.2f}")
    log(f"Files are in {path('output', '')}")


# ------------------------------------------------------------------ main
STAGES = {
    "prep": lambda: (stage_prep("train"), stage_prep("test")),
    "index": lambda: (stage_index("train"), stage_index("test")),
    "trainfe": stage_trainfe,
    "train": stage_train,
    "testfe": stage_testfe,
    "write": stage_write,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"] + list(STAGES))
    ap.add_argument("--data", default=DATA_DIR, help="folder containing train/ and test/")
    ap.add_argument("--work", default=WORK_DIR, help="folder where checkpoints and outputs are saved")
    ap.add_argument("--prep-root", default=PREP_ROOT, help="optional: reuse cleaned data from this folder")
    ap.add_argument("--index-root", default=INDEX_ROOT, help="optional: reuse blocking index from this folder")
    args = ap.parse_args()
    DATA_DIR, WORK_DIR = args.data, args.work
    PREP_ROOT, INDEX_ROOT = args.prep_root, args.index_root
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        log(f"===== stage: {name} =====")
        STAGES[name]()
    log("done")
