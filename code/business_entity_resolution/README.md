# Business Entity Resolution — Team Jishnu (Amazon ML Challenge 2026)

Final pipeline (internally "v7"): public leaderboard F0.5 = **0.920**; stage-2 cross-validated F0.5 = 0.933; **3.72 candidates per Source 1 entity**.

## Environment
- Google Colab free tier, CPU runtime (2 vCPU, ~12 GB RAM). No GPU needed.
- `pip install -r requirements.txt`
- Only the provided challenge data is used. No external data, APIs or pretrained models.

## Reproduce end to end (data -> blocking -> matching -> output)
```bash
python src/pipeline.py --stage all \
    --data /path/to/student_resource/dataset \
    --work /path/to/work_folder
```
Step 2: learned pruning + stage-2 final matcher (about 30-40 minutes):
```bash
JISHNU_WORK=/path/to/work_folder JISHNU_DATA=/path/to/student_resource/dataset \
    python src/pruned_stage2.py
```
Final outputs: `<work_folder>/final/matching_results.tsv` and `<work_folder>/final/candidate_pairs.tsv`
(step 1 alone writes the stage-1 outputs to `<work_folder>/output/`).

Total runtime from scratch on free Colab is about 8 hours:
prep ~10 min, index ~75 min, trainfe ~35 min, train ~50 min, testfe ~5 h, write ~5 min.

## Stages (each can be run alone with --stage <name>)
| Stage | What it does | Saved to `<work>/` |
|---|---|---|
| prep | Normalize names and addresses (lowercase, accents, abbreviations, legal suffixes) | `prep/*.parquet` |
| index | Build hashed blocking-key index for Source 2 + Source 3 (train and test) | `index/*.npy` |
| trainfe | Candidates + 36 pair features for a 250k sample of training Source 1 entities | `trainfe/*.parquet` |
| train | LightGBM, checkpoint every 100 trees; tune decision thresholds on a 20% validation split | `model/` |
| testfe | Candidates + features + probabilities for all test Source 1 entities, 50k per chunk | `testpred/*.parquet` |
| write | One-owner rule + thresholds -> both output TSV files | `output/` |

## Resuming after a timeout
Every stage and every chunk is saved as it finishes (atomic writes). Re-running the same command
skips finished work and continues from the last saved chunk or model checkpoint.

## Validate
```bash
python3 utils/validate_submission.py --matching <work>/output/matching_results.tsv \
    --candidate <work>/output/candidate_pairs.tsv --test-dir dataset/test
```

## Notes
- The competition run reused cleaned data and the blocking index from earlier runs of identical code
  (`--prep-root`, `--index-root`). A fresh run with both flags left empty rebuilds everything and
  produces the same result (fixed random seed 42).
- Models: LightGBM (MIT licence), well under the 8B-parameter limit. String similarity: RapidFuzz (MIT).
