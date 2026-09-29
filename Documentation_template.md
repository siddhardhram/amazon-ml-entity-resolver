# ML Challenge 2026: Business Entity Resolution Solution Template

**Team:** Jishnu

## 1. Executive Summary
We built a scalable two-stage entity-resolution pipeline that runs entirely on a free Google Colab CPU
runtime: text normalization, hashed multi-key blocking with rarity-weighted candidate ranking, and a
LightGBM pair classifier on 36 similarity features, followed by F0.5-tuned decision rules and a
"one-owner" constraint discovered in the training ground truth.

**Final result (v7):** public leaderboard F0.5 = **0.920**; stage-2 cross-validated F0.5 = 0.933; **3.72 candidates per Source 1 entity** in the final candidate set (key blocking alone: ≈48). Best stage-1-only result (v6): leaderboard 0.919.
The score rose from 0.762 (first baseline) through five diagnostic-driven iterations; the largest gains
came from blocking improvements. Only the provided data is used; the only model is LightGBM (MIT).

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** train has 2.2M Source 1 (S1) and 10.3M Source 2+3 (S2/S3) records; test has 1.73M S1 and
  10.0M S2/S3 records. All-pairs comparison (~10^13 pairs) is impossible, so blocking is essential.
- **One-owner property:** in the training ground truth, 0 of 7,638,365 linked S2/S3 records belong to
  more than one S1 entity. Each S2/S3 record therefore matches at most one S1.
- **Match counts:** 5.6% of S1 entities are singletons; most have 2–5 matches (maximum 11).
  About 2.7M training S2/S3 records match no S1 at all (distractors).
- **Country:** true matches always share the country label (0% cross-country pairs among misses), so
  blocking is done within each country. Country is treated as an open set of labels: France appears
  only in test (15% of test S1) and nothing is hard-coded to US/India.
- **Noise:** abbreviations, legal suffixes, typos, transliterations (Lakshmi/Laxmi), digit-letter swaps
  (r0yce), concatenated names, `.com` trade names, empty names, and partial or reordered addresses.
  Many Indian S2/S3 records have an empty or unrelated name but a matching address.
- **Metric:** macro F0.5 per S1 entity, singletons included: a correct empty prediction earns 1.0 and any
  false match on a singleton earns 0, so precision is favoured.

### 2.2 Solution Strategy
1. **Normalization:** Unicode NFKD accent stripping (for unseen French data), lowercasing, `&`→`and`,
   punctuation removal, abbreviation expansion (pvt→private, rd→road, …, French av→avenue,
   bd→boulevard, r→rue, pl→place), a **core name** without legal suffixes (private, limited, llc, inc,
   sarl, sas, gmbh, …), and a **phonetic code** per token (consonant skeleton after rules such as
   ksh→x, ph→f, w→v, z→s, doubled-letter collapse, digit→letter fixes inside words).
2. **Blocking** with hashed keys and rarity-weighted ranking (Section 3).
3. **Pair classification** with LightGBM (Section 4).
4. **Decisions** tuned for F0.5 plus the one-owner rule.
5. **Engineering:** every stage writes chunked, atomic checkpoints (50k S1 per chunk, a model checkpoint
   every 100 trees), so the ~8-hour run resumes after Colab timeouts.
6. **Diagnostic-driven iteration:** after each version a diagnostic tool classified every missed true
   match as "no shared key", "key too frequent" or "cut by candidate ranking", and computed the F0.5
   ceiling a perfect model would reach on our candidates. Each next version targeted the largest category.

## 3. Candidate Generation (Blocking)
Each record emits a set of blocking keys, always prefixed with its country. Keys are hashed to uint64;
all S2/S3 keys are stored in one sorted array, so lookups are vectorized binary searches.

**Name keys:** rare tokens; phonetic tokens; sorted phonetic token pairs among the first four tokens
(robust to word order and typos); 4-character prefixes of the first two tokens (in order and sorted);
the concatenated name ("solutionshimanshu" = "solutions himanshu").

**Address keys:** rare locality words (≥5 letters, minus a stop-list of generic words), pairs of
numeric tokens with leading zeros removed ("09 4 275" = "9 4 275"), number + following street word
("51 church"), consecutive word pairs, and address number + name prefix. These find matches even when
the S2/S3 name is empty or unrelated.

**Frequency cap:** keys shared by more than 200 records are dropped (e.g. "restaurant", "mumbai").

**Candidate ranking:** candidates are scored by **rarity-weighted shared keys** (each key weighted by
log(1 + 200 / block size)); the top 150 are re-scored with rarity + name token-set similarity + address
token-set similarity (an empty name counts as neutral, not as a mismatch); the **top 50** are kept.
**Learned pruning (meta-blocking).** The ≈48 key-blocked candidates are scored by the stage-1 LightGBM (Section 4). Only candidates with stage-1 probability ≥ τ = 0.20 are kept, which cuts the set to **3.72 candidates per S1 on test**; recall over all true matches goes from 0.938 to 0.920 on validation (98.1% of the matches found by key blocking are kept). τ was chosen as the largest value whose stage-2 cross-validated F0.5 stayed within 0.001 of the best:

| τ | Candidates per S1 | Recall (all true matches) | Stage-2 CV F0.5 |
|---|---|---|---|
| 0.01 | 4.66 | 0.936 | 0.9331 |
| 0.02 | 4.33 | 0.935 | 0.9332 |
| 0.05 | 3.97 | 0.932 | 0.9332 |
| 0.10 | 3.73 | 0.928 | 0.9331 |
| 0.15 | 3.61 | 0.924 | 0.9331 |
| 0.20 | 3.52 | 0.920 | 0.9332 |

`candidate_pairs.tsv` is exactly this pruned set, the only input to the final stage-2 model.

| Version | Blocking change | Recall (train sample) |
|---|---|---|
| v1 | Rare name tokens, name prefixes, address number + name start; top 30 | 0.712 |
| v3 | + phonetic tokens, sorted phonetic pairs | 0.802 |
| v5 | + address keys, name+address pre-ranking; top 40 | 0.904 |
| v6 | + rarity-weighted ranking; top 50 | 0.938 |
| v7 | + learned pruning (stage-1 probability ≥ 0.20); 3.72 candidates per S1 | 0.920 |

Final candidates per S1: **3.72** on test out of ~10M S2/S3 records (key blocking alone: ≈48).
Only 0.4% of test S1 entities have no candidates (v1: 10.8%).

## 4. Matching Model
**Model:** LightGBM binary classifier (MIT licence), 600 trees, learning rate 0.05, 63 leaves, min 50
samples per leaf, feature and bagging fraction 0.8, seed 42. Training data: all candidates of a random
250,000 training S1 entities (≈12M pairs), labels from the ground truth. 20% of these entities are held
out (split by entity, never by pair) for validation with the official macro F0.5, singletons included.

**36 features per (S1, candidate) pair:**
- *Name:* ratio, token-sort, token-set and partial ratio (RapidFuzz) on normalized and core names; token
  Jaccard; first-token equality; length difference; no-space ratio; acronym match ("fsp" = "family
  select physicians"); name containment; empty-name flags.
- *Phonetic:* token-set and ratio on phonetic names; first phonetic token equality.
- *Address:* ratio, token-set, partial ratio; numeric-token Jaccard and overlap; shared rare address
  words; containment (share of the shorter address's words found in the longer one).
- *Blocking evidence:* number of shared keys; rarity-weighted shared keys.
- *Relative (within each S1's candidate list):* rank and gap-to-best for name, address, phonetic and
  rarity scores; number of candidates. These help pick the right firm when many firms share a building.
- *Source flag:* S2 or S3.

**Stage-2 final matcher (v7):** a second LightGBM (300 trees, 31 leaves, MIT licence) runs only on the pruned candidates. Its 14 features describe each candidate's stage-1 probability in the context of its S1's full key-blocked list: probability, rank, best and second-best probability, gap to the best and to the next, number of candidates above 0.5 and 0.3, probability sum and share, list size, source flag, and whether it is the best candidate of its source. It is trained on the stage-1 validation entities, whose probabilities are out-of-sample like the test set, and assessed with 5-fold cross-validation by entity. A candidate is matched if its stage-2 probability ≥ 0.65, followed by the one-owner rule (73,755 conflicting claims removed).

**Stage-1 decision rules (v6, for reference):** keep a candidate if probability ≥ 0.65 (tuned on validation; separate thresholds for
each S1's top candidate were also searched and converged to the same value). Then the **one-owner rule**
assigns each S2/S3 record only to the S1 with the highest probability (104,431 conflicting claims
removed in the final run).

## 5. Results & Error Analysis
| Version | Main change | Validation F0.5 | Leaderboard F0.5 |
|---|---|---|---|
| v1 | Baseline: normalization, name keys, LightGBM, 21 features | 0.801 | 0.762 |
| v3 | Phonetic keys, French normalization, phonetic features | 0.855 | 0.832 |
| v4 | + one-owner rule (post-processing) | – | 0.838 |
| v5 | Address blocking, name+address pre-ranking, 5 new features | 0.915 | 0.903 |
| v6 | Rarity-weighted ranking, top 50, 250k training entities, 5 new features | 0.932 | 0.919 |
| v7 | Learned pruning to 3.72 candidates/S1 + stage-2 matcher | 0.933 (CV) | **0.920** |

**Error analysis (diagnostics on the training sample):**
- v3: ceiling 0.894; India recall 0.68 vs US 0.89. Misses: 48% cut by ranking, 27% no shared key,
  25% key too frequent. Indian S2/S3 records often had empty names but matching rare address words
  → address keys (v5).
- v5: ceiling 0.957; India 0.845, US 0.945. "No shared key" fell to 0.6%; 60% of misses were found but
  cut by ranking, typically empty-name records with short addresses outranked by other firms in the
  same building → rarity-weighted ranking and neutral empty names (v6).
- The validation–leaderboard gap shrank from 0.039 (v1) to 0.013 (v6), most likely thanks to French
  normalization, since France is absent from training.

**Tried and rejected:** per-entity expected-F0.5 decoding from predicted probabilities scored 0.914 on
validation vs 0.932 for tuned thresholds, so thresholds were kept.

## 6. Conclusion
Blocking quality was the dominant factor: each recall gain translated almost directly into score, and
diagnosing *why* matches were missed was more valuable than model tuning. With v6 candidates a perfect
model would score about 0.96, so next steps would be model-side (e.g. two-stage re-ranking with
group-level features, a larger training sample) and reducing the remaining ~6% of blocking misses,
mostly caused by very common keys. The full pipeline runs on free-tier hardware and is resumable.

**Compliance:** only the provided training and test data was used. No external databases, APIs,
geocoding or internet data; no pretrained language models. The only model is LightGBM (MIT licence).

## Appendix

### A. Code Artefacts
- `code/business_entity_resolution/src/pipeline.py`: the complete pipeline, with stages
  prep → index → trainfe → train → testfe → write; each can be run alone with `--stage`.
- `code/business_entity_resolution/README.md`: exact reproduction and resume instructions.
- `code/business_entity_resolution/requirements.txt`: pinned package versions.
- `code/business_entity_resolution/src/pruned_stage2.py`: learned pruning + stage-2 final matcher (run after pipeline.py).
- `output/matching_results.tsv`: final matches (the submitted leaderboard file).
- `output/candidate_pairs.tsv`: final (pruned) candidate set fed to the stage-2 model.

Reproduce: `python src/pipeline.py --stage all --data <dataset folder> --work <work folder>`
(about 8 hours on a free Colab CPU runtime).

### B. Additional Results
| Metric | v1 | v3 | v5 | v6 |
|---|---|---|---|---|
| Blocking recall (train sample) | 0.712 | 0.802 | 0.904 | 0.938 |
| Perfect-model ceiling on candidates | – | 0.894 | 0.957 | – |
| Average candidates per S1 | 18.8 | 27.7 | 38.5 | 47.7 |
| Test S1 predicted as no-match | 21.0% | 13.2% | 8.2% | 7.4% |
| Test S1 with zero candidates | 10.8% | – | 0.4% | – |

Test composition: 809,986 India, 663,106 US and 259,452 France S1 entities.
