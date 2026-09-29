"""
Submission Validator & Statistics Checker
Amazon ML Challenge 2026 - Business Entity Resolution (Team Jishnu)

Validates:
1. TSV header and formatting
2. Row-by-row alignment between candidate_pairs.tsv and matching_results.tsv
3. Entity ID syntax (S1-xxx, S2-xxx, S3-xxx)
4. Candidate subset integrity: all matched entities MUST be in candidate_pairs
5. One-owner constraint: no S2/S3 entity matched to more than one S1
6. Candidate & match distributions, singleton ratios, and performance metrics
"""

import sys
import os
import time
import re
from collections import Counter

S1_PATTERN = re.compile(r"^S1-\d+$")
CAND_PATTERN = re.compile(r"^(S2|S3)-\d+$")

def format_num(n):
    return f"{n:,}"

def validate(candidate_path, matching_path):
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026: SUBMISSION VALIDATION & DIAGNOSTICS")
    print("=" * 70)
    
    if not os.path.exists(candidate_path):
        print(f"[ERROR] Candidate pairs file not found: {candidate_path}")
        return False
    if not os.path.exists(matching_path):
        print(f"[ERROR] Matching results file not found: {matching_path}")
        return False
        
    cand_size_mb = os.path.getsize(candidate_path) / (1024 * 1024)
    match_size_mb = os.path.getsize(matching_path) / (1024 * 1024)
    print(f"Candidate file : {candidate_path} ({cand_size_mb:.2f} MB)")
    print(f"Matching file  : {matching_path} ({match_size_mb:.2f} MB)")
    print("-" * 70)

    t0 = time.time()
    
    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    
    s1_with_no_candidates = 0
    s1_with_no_matches = 0
    
    cand_count_dist = Counter()
    match_count_dist = Counter()
    
    s2_matches = 0
    s3_matches = 0
    s2_candidates = 0
    s3_candidates = 0
    
    s2_s3_owners = {}
    one_owner_violations = 0
    subset_violations = 0
    id_format_errors = 0
    order_mismatches = 0
    
    max_reported_errors = 5
    reported_errors = []

    print("Streaming files and validating line by line...")
    
    with open(candidate_path, "r", encoding="utf-8") as f_cand, \
         open(matching_path, "r", encoding="utf-8") as f_match:
        
        cand_header = f_cand.readline().rstrip("\r\n")
        match_header = f_match.readline().rstrip("\r\n")
        
        expected_cand_header = "source1_entity_id\tcandidate_entity_ids"
        expected_match_header = "source1_entity_id\tmatched_entity_ids"
        
        if cand_header != expected_cand_header:
            reported_errors.append(f"Invalid candidate header: '{cand_header}' (expected '{expected_cand_header}')")
        if match_header != expected_match_header:
            reported_errors.append(f"Invalid matching header: '{match_header}' (expected '{expected_match_header}')")
            
        line_num = 1
        seen_s1 = set()
        
        for cand_line, match_line in zip(f_cand, f_match):
            line_num += 1
            total_s1 += 1
            
            c_parts = cand_line.rstrip("\r\n").split("\t")
            m_parts = match_line.rstrip("\r\n").split("\t")
            
            if len(c_parts) != 2:
                if len(reported_errors) < max_reported_errors:
                    reported_errors.append(f"Line {line_num}: candidate_pairs has {len(c_parts)} columns (expected 2)")
                continue
            if len(m_parts) != 2:
                if len(reported_errors) < max_reported_errors:
                    reported_errors.append(f"Line {line_num}: matching_results has {len(m_parts)} columns (expected 2)")
                continue
                
            s1_cand, cands_str = c_parts[0], c_parts[1]
            s1_match, matches_str = m_parts[0], m_parts[1]
            
            if s1_cand != s1_match:
                order_mismatches += 1
                if len(reported_errors) < max_reported_errors:
                    reported_errors.append(f"Line {line_num}: S1 ID mismatch: cand='{s1_cand}' vs match='{s1_match}'")
                continue
                
            if not S1_PATTERN.match(s1_cand):
                id_format_errors += 1
                if len(reported_errors) < max_reported_errors:
                    reported_errors.append(f"Line {line_num}: Invalid S1 format: '{s1_cand}'")
                    
            if s1_cand in seen_s1:
                if len(reported_errors) < max_reported_errors:
                    reported_errors.append(f"Line {line_num}: Duplicate S1 ID detected: '{s1_cand}'")
            seen_s1.add(s1_cand)
            
            cands = cands_str.split(",") if cands_str else []
            cand_set = set(cands)
            total_candidates += len(cands)
            cand_count_dist[len(cands)] += 1
            if not cands:
                s1_with_no_candidates += 1
                
            for c in cands:
                if not CAND_PATTERN.match(c):
                    id_format_errors += 1
                    if len(reported_errors) < max_reported_errors:
                        reported_errors.append(f"Line {line_num}: Invalid candidate ID format: '{c}'")
                if c.startswith("S2"):
                    s2_candidates += 1
                elif c.startswith("S3"):
                    s3_candidates += 1
                    
            matches = matches_str.split(",") if matches_str else []
            total_matches += len(matches)
            match_count_dist[len(matches)] += 1
            if not matches:
                s1_with_no_matches += 1
                
            for m in matches:
                if not CAND_PATTERN.match(m):
                    id_format_errors += 1
                    if len(reported_errors) < max_reported_errors:
                        reported_errors.append(f"Line {line_num}: Invalid match ID format: '{m}'")
                        
                if m not in cand_set:
                    subset_violations += 1
                    if len(reported_errors) < max_reported_errors:
                        reported_errors.append(f"Line {line_num}: Match '{m}' not present in candidate list for '{s1_cand}'")
                        
                prev_owner = s2_s3_owners.get(m)
                if prev_owner is not None and prev_owner != s1_cand:
                    one_owner_violations += 1
                    if len(reported_errors) < max_reported_errors:
                        reported_errors.append(f"Line {line_num}: One-owner violation: '{m}' assigned to both '{prev_owner}' and '{s1_cand}'")
                else:
                    s2_s3_owners[m] = s1_cand
                    
                if m.startswith("S2"):
                    s2_matches += 1
                elif m.startswith("S3"):
                    s3_matches += 1
                    
            if total_s1 % 500_000 == 0:
                print(f"  Processed {format_num(total_s1)} entities...")

        rem_c = len(f_cand.readlines())
        rem_m = len(f_match.readlines())
        if rem_c > 0 or rem_m > 0:
            order_mismatches += 1
            reported_errors.append(f"File line count mismatch: cand has {rem_c} extra lines, match has {rem_m} extra lines")

    elapsed = time.time() - t0
    
    print("-" * 70)
    print(f"Validation completed in {elapsed:.2f} seconds.")
    print("=" * 70)
    print("VALIDATION SUMMARY & DIAGNOSTICS")
    print("=" * 70)
    print(f"Total Test S1 Entities         : {format_num(total_s1)}")
    print(f"Total Candidate Pairs           : {format_num(total_candidates)} (avg {total_candidates/max(1, total_s1):.2f} / S1)")
    print(f"Total Predicted Matches         : {format_num(total_matches)} (avg {total_matches/max(1, total_s1):.2f} / S1)")
    print(f"S1 with 0 candidates            : {format_num(s1_with_no_candidates)} ({s1_with_no_candidates/max(1, total_s1):.2%})")
    print(f"S1 with 0 predicted matches     : {format_num(s1_with_no_matches)} ({s1_with_no_matches/max(1, total_s1):.2%})")
    print(f"Unique matched S2/S3 entities   : {format_num(len(s2_s3_owners))}")
    print()
    print("Source Breakdown:")
    print(f"  Candidates : S2 = {format_num(s2_candidates)} ({s2_candidates/max(1, total_candidates):.1%}), S3 = {format_num(s3_candidates)} ({s3_candidates/max(1, total_candidates):.1%})")
    print(f"  Matches    : S2 = {format_num(s2_matches)} ({s2_matches/max(1, total_matches):.1%}), S3 = {format_num(s3_matches)} ({s3_matches/max(1, total_matches):.1%})")
    print()
    print("Integrity & Constraint Checks:")
    print(f"  [✓] Line Count & S1 Alignment   : {'PASSED' if order_mismatches == 0 else f'FAILED ({order_mismatches} mismatches)'}")
    print(f"  [✓] ID Format Verification     : {'PASSED' if id_format_errors == 0 else f'FAILED ({id_format_errors} errors)'}")
    print(f"  [✓] Candidate Subset Property   : {'PASSED' if subset_violations == 0 else f'FAILED ({subset_violations} violations)'}")
    print(f"  [✓] One-Owner Rule Constraint   : {'PASSED' if one_owner_violations == 0 else f'FAILED ({one_owner_violations} conflicts)'}")
    print()
    print("Match Count Distribution per S1:")
    for k in sorted(match_count_dist.keys())[:10]:
        print(f"  {k} matches: {format_num(match_count_dist[k]):>10} entities ({match_count_dist[k]/max(1, total_s1):.2%})")
    if any(k >= 10 for k in match_count_dist):
        ge_10 = sum(v for k, v in match_count_dist.items() if k >= 10)
        print(f"  >=10 matches: {format_num(ge_10):>8} entities ({ge_10/max(1, total_s1):.2%})")

    print("-" * 70)
    has_errors = (order_mismatches > 0 or id_format_errors > 0 or 
                  subset_violations > 0 or one_owner_violations > 0 or 
                  len(reported_errors) > 0)
                  
    if not has_errors:
        print("[SUCCESS] All validation and integrity checks PASSED without any errors!")
        print("The submission files are completely valid, consistent, and optimally generated.")
        return True
    else:
        print("[FAILURE] Errors or integrity violations were detected:")
        for err in reported_errors:
            print(f"  - {err}")
        return False

if __name__ == "__main__":
    cand_file = os.path.abspath("output/candidate_pairs.tsv")
    match_file = os.path.abspath("output/matching_results.tsv")
    
    if len(sys.argv) >= 3:
        cand_file = sys.argv[1]
        match_file = sys.argv[2]
        
    success = validate(cand_file, match_file)
    sys.exit(0 if success else 1)
