# Amazon ML Challenge 2026: Business Entity Resolution (Team Jishnu)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Leaderboard F0.5](https://img.shields.io/badge/Public%20Leaderboard%20F0.5-0.920-brightgreen.svg)]()
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-orange.svg)]()

High-performance, scalable entity resolution pipeline built for the Amazon ML Challenge 2026. Resolves matches across Source 1 (S1) and Source 2/3 (S2, S3) entities at scale (millions of records) under CPU constraints.

---

## 🏆 Key Results (Pipeline v7)
- **Public Leaderboard F0.5:** **0.920**
- **Stage-2 5-fold CV F0.5:** **0.933**
- **Candidate Efficiency:** Reduced from ~48 to **3.72 candidates per S1 entity** via learned pruning (meta-blocking).
- **One-Owner Property:** Enforces strict one-owner matching discovered in the training ground truth.

---

## 📁 Repository Structure

```text
├── Documentation_template.md        # Complete solution report and methodology
├── .gitattributes                   # Git LFS tracking configuration for large TSVs
├── .gitignore                       # Ignored files and temporary cache
├── code/
│   └── business_entity_resolution/
│       ├── README.md                # Execution guide & Colab reproduction steps
│       ├── requirements.txt         # Dependencies (lightgbm, rapidfuzz, pandas, pyarrow)
│       ├── validate_submission.py   # Fast streaming submission validator & sanity checker
│       └── src/
│           ├── pipeline.py          # Stage 1: Prep, multi-key blocking, feature extraction & LGBM
│           └── pruned_stage2.py     # Stage 2: Learned pruning (tau=0.20) & final matcher
└── output/                          # Tracked with Git LFS
    ├── candidate_pairs.tsv          # Pruned candidate pairs (1,732,545 test entities, 105.5 MB)
    └── matching_results.tsv         # Final matched entity pairs (1,732,545 test entities, 90.7 MB)
```

---

## 🚀 Quick Start

### 1. Validate Existing Submissions
Validate format integrity, candidate subset consistency, and the one-owner constraint across all 1.73M test records:
```bash
python code/business_entity_resolution/validate_submission.py
```

### 2. End-to-End Pipeline Execution (From Scratch)
Install dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Run Stage 1 (prep -> blocking index -> feature extraction -> LightGBM):
```bash
python code/business_entity_resolution/src/pipeline.py --stage all \
    --data /path/to/dataset \
    --work /path/to/work_folder
```

Run Stage 2 (learned pruning + final matcher):
```bash
JISHNU_WORK=/path/to/work_folder JISHNU_DATA=/path/to/dataset \
    python code/business_entity_resolution/src/pruned_stage2.py
```

---

## 🧠 Methodology Summary
1. **Normalization:** Unicode NFKD accent stripping, abbreviation expansion, legal suffix stripping, and phonetic token coding.
2. **Hashed Multi-Key Blocking:** Rare name tokens, phonetic token pairs, address locality words, consecutive word pairs, and address number + name prefix.
3. **Candidate Ranking & Pruning:** Candidates ranked by rarity-weighted shared keys and pruned via Stage-1 probability threshold ($\tau = 0.20$).
4. **Matching & Constraints:** LightGBM pair classification with 36 features, followed by global one-owner conflict resolution.
