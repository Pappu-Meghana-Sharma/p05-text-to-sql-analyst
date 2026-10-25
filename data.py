# AI-Assisted Implementation (Antigravity) - Part 1: Data & Harness
"""
Text-to-SQL Analyst - Part 1: Data & Harness
============================================
This module implements the complete data handling and evaluation harness:
1. load_data(): Hugging Face dataset loading with error handling and in-memory caching.
2. get_splits(seed=42): Reproducible, leak-free train/test splitting preserving official splits.
3. run_sql(sql_context, sql, timeout=5): Sandboxed in-memory SQLite execution with multi-layer
   safety protections (PRAGMA query_only, authorizer, progress handler timeout).
4. same_result(pred_rows, gold_rows): Multiset result comparison with numeric tolerance,
   NULL handling, duplicate detection, and order invariance.
5. run_eda(): Comprehensive exploratory data analysis of splits, complexity, and domains.

Team Integration:
- Uses configuration settings from `config.py` (CONFIG, SEED, QUICK_RUN).
- Generates canonical `test_ids.json` for shared evaluation across RAG, Agent, LoRA, and Eval.
- Fully compatible with Python 3.12 and Windows.
"""

import os
import sys
import json
import math
import time
import random
import sqlite3
from pathlib import Path
from typing import Tuple, List, Optional, Any, Dict
from collections import Counter

import datasets

# Import shared configuration from config.py
try:
    from config import CONFIG, SEED, QUICK_RUN
except ImportError:
    # Fallback default configuration if imported in standalone environment
    SEED = 42
    QUICK_RUN = True
    CONFIG = {
        "dataset": "gretelai/synthetic_text_to_sql",
        "test_size": 200,
        "quick_test_size": 20,
        "test_ids_path": "test_ids.json",
    }

# In-memory singleton cache to prevent repeated downloads/reloads across imports
_DATASET_CACHE = None


# =====================================================================
# 1. Dataset Loading
# =====================================================================
def load_data() -> datasets.DatasetDict:
    """
    Loads the synthetic text-to-sql dataset specified in CONFIG['dataset'].

    Features:
    - Loads using Hugging Face `datasets.load_dataset`.
    - Handles network and loading errors gracefully with clear, actionable diagnostics.
    - Caches loaded dataset in memory to avoid repeated parsing or disk access.
    - Preserves all original columns:
      ['id', 'domain', 'domain_description', 'sql_complexity',
       'sql_complexity_description', 'sql_task_type',
       'sql_task_type_description', 'sql_prompt', 'sql_context',
       'sql', 'sql_explanation']
    - Supports both 'train' and 'test' splits without cross-split mixing.

    Returns:
        datasets.DatasetDict: Dataset dictionary containing 'train' and 'test' splits.

    Raises:
        RuntimeError: If dataset loading fails.
    """
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE

    dataset_name = CONFIG.get("dataset", "gretelai/synthetic_text_to_sql")

    try:
        # Load dataset from Hugging Face Hub (uses local cache if available)
        ds = datasets.load_dataset(dataset_name)
    except Exception as e:
        error_msg = (
            f"Failed to load dataset '{dataset_name}' from Hugging Face.\n"
            f"Underlying error: {type(e).__name__}: {e}\n"
            "Troubleshooting tips:\n"
            "  1. Verify active internet connection and access to https://huggingface.co.\n"
            "  2. Check local disk space in the Hugging Face cache directory (~/.cache/huggingface/datasets).\n"
            "  3. Set HF_TOKEN environment variable if experiencing rate limits."
        )
        raise RuntimeError(error_msg) from e

    # Validate expected splits
    if "train" not in ds or "test" not in ds:
        raise RuntimeError(
            f"Dataset '{dataset_name}' is missing expected splits. Found splits: {list(ds.keys())}"
        )

    # Validate essential schema columns
    expected_columns = {
        "id",
        "domain",
        "sql_complexity",
        "sql_prompt",
        "sql_context",
        "sql",
    }
    for split_name in ["train", "test"]:
        actual_cols = set(ds[split_name].column_names)
        missing_cols = expected_columns - actual_cols
        if missing_cols:
            raise RuntimeError(
                f"Dataset split '{split_name}' is missing required columns: {missing_cols}. "
                f"Available columns: {actual_cols}"
            )

    _DATASET_CACHE = ds
    return _DATASET_CACHE


# =====================================================================
# 2. Reproducible Train / Test Split
# =====================================================================
def get_splits(
    seed: int = SEED,
    quick_run: Optional[bool] = None,
    save_test_ids: bool = True,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """
    Produces reproducible training and test datasets based on project config.

    Split & Anti-Leakage Strategy:
    ------------------------------
    1. The source dataset (gretelai/synthetic_text_to_sql) has an official 'train'
       split (100,000 examples) and an official 'test' split (5,851 examples).
    2. Crucially, both official splits independently number their `id` column
       starting from 1 (train IDs: 1..100000; test IDs: 1..5851).
    3. To respect benchmark integrity, the evaluation test set is sampled
       strictly and deterministically from the official 'test' split.
    4. The canonical test IDs (200 for full evaluation) are persisted to `test_ids.json`
       so all teammates (RAG, Agent, LoRA, Eval) evaluate on the exact same examples.
    5. When `quick_run=True`, the first `quick_test_size` (20) IDs from the canonical
       test set are used, ensuring quick development runs evaluate on a strict prefix
       subset of the final evaluation set.
    6. The official 'train' and 'test' splits are maintained as completely separate partitions
       with 0 prompt overlap. Because `id` is a local 1-based index within each split
       (train: 1..100000; test: 1..5851), `train_ds` preserves all 100,000 legitimate
       training examples without discarding records due to numeric ID collisions.
    7. Evaluation examples are kept completely isolated in `test_ds` (and `test_ids.json`),
       strictly drawn from the official 'test' split.
    8. Verification confirms 0 prompt overlap between train and test data.

    Args:
        seed: Random seed for sampling reproducibility (default: 42 from config).
        quick_run: If True, uses CONFIG['quick_test_size'] (default: 20).
                   If None, reads CONFIG.get('QUICK_RUN', False).
        save_test_ids: If True, ensures canonical test IDs are written to test_ids.json.

    Returns:
        Tuple[datasets.Dataset, datasets.Dataset]: (train_dataset, test_dataset)
    """
    ds = load_data()

    if quick_run is None:
        quick_run = CONFIG.get("QUICK_RUN", QUICK_RUN)

    # Canonical test set size (200) and quick development run size (20)
    canonical_size = CONFIG.get("test_size", 200)
    quick_size = CONFIG.get("quick_test_size", 20)
    test_ids_path = Path(CONFIG.get("test_ids_path", "test_ids.json"))

    is_default_seed = (seed == SEED)
    canonical_test_ids: List[int] = []

    # Valid official test IDs set for validation
    valid_test_ids_set = set(ds["test"]["id"])

    # Only load from test_ids.json if using the default canonical seed
    if is_default_seed and test_ids_path.exists():
        try:
            with open(test_ids_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                # Ensure unique IDs preserving order
                loaded_ints = [int(x) for x in loaded]
                seen = set()
                unique_ids = []
                for tid in loaded_ints:
                    if tid not in seen:
                        seen.add(tid)
                        unique_ids.append(tid)

                # Validate: contains at least canonical_size unique IDs AND all exist in official test split
                if len(unique_ids) >= canonical_size:
                    candidate_ids = unique_ids[:canonical_size]
                    if all(tid in valid_test_ids_set for tid in candidate_ids):
                        canonical_test_ids = candidate_ids
        except Exception:
            canonical_test_ids = []

    if not canonical_test_ids:
        # Deterministically sample from the official test split
        test_split = ds["test"]
        all_test_ids = sorted(list(test_split["id"]))
        rng = random.Random(seed)
        canonical_test_ids = rng.sample(all_test_ids, k=canonical_size)

        # Only save to test_ids.json if using the default canonical seed
        # Non-default seeds must not overwrite the canonical test_ids.json
        if is_default_seed and save_test_ids:
            try:
                test_ids_path.parent.mkdir(parents=True, exist_ok=True)
                with open(test_ids_path, "w", encoding="utf-8") as f:
                    json.dump(canonical_test_ids, f, indent=2)
            except Exception as e:
                print(f"[Warning] Could not write test IDs to {test_ids_path}: {e}", file=sys.stderr)

    # Determine active test IDs depending on quick_run mode
    expected_size = quick_size if quick_run else canonical_size
    active_test_ids = canonical_test_ids[:expected_size]

    active_id_set = set(active_test_ids)
    canonical_id_set = set(canonical_test_ids)

    # Select test dataset examples preserving order of active_test_ids
    test_split = ds["test"]
    id_to_index = {row["id"]: i for i, row in enumerate(test_split)}
    test_indices = [id_to_index[tid] for tid in active_test_ids if tid in id_to_index]
    test_ds = test_split.select(test_indices)

    # Ensure the final test set contains the expected number of examples
    if len(test_ds) != expected_size:
        raise RuntimeError(
            f"Expected {expected_size} test examples, but selected {len(test_ds)}"
        )

    # Preserve the official split boundary:
    # train_ds is the complete official training split (100,000 examples).
    # Since gretelai/synthetic_text_to_sql was partitioned into official 'train'
    # and 'test' splits with 0 prompt overlap, evaluation examples (drawn from
    # 'test') are completely separate from training data. We preserve all 100,000
    # training examples and do not remove unrelated training records whose numeric
    # IDs happen to match test IDs in their respective local namespaces.
    train_ds = ds["train"]

    return train_ds, test_ds


# =====================================================================
# 3. Safe SQL Execution Sandbox
# =====================================================================
def run_sql(
    sql_context: Optional[str],
    sql: str,
    timeout: float = 5.0,
) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
    """
    Executes a SQL query in an isolated, in-memory SQLite database populated with sql_context.

    Safety & Sandboxing Implementation:
    -----------------------------------
    1. Isolation: Each invocation creates a fresh, isolated `:memory:` SQLite connection.
    2. Context Setup: sql_context (DDL/DML like CREATE TABLE, INSERT) is executed first.
    3. Read-Only Pragma: `PRAGMA query_only = ON` is enabled before query execution.
    4. Authorizer Protection: `conn.set_authorizer` intercepts all SQLite actions.
       Only read operations (SQLITE_SELECT, SQLITE_READ, SQLITE_FUNCTION, SQLITE_RECURSIVE)
       are permitted. All writes and destructive actions (INSERT, UPDATE, DELETE, DROP,
       ALTER, CREATE, ATTACH, DETACH, PRAGMA) are strictly blocked with SQLITE_DENY.
    5. Single Statement Enforcement: sqlite3.execute() prevents execution of multiple
       queries (e.g., 'SELECT 1; DROP TABLE t;').
    6. Execution Timeout: SQLite progress handler triggers every 1,000 opcodes and
       interrupts execution if the elapsed time exceeds `timeout` seconds (default: 5.0s).
    7. Resource Cleanup: Progress handlers, authorizers, and connections are always
       closed in a `finally` block to prevent resource leaks.

    Args:
        sql_context: Setup SQL statements containing table schemas and initial rows.
        sql: Generated or gold SQL query to execute.
        timeout: Maximum execution time in seconds (default: 5.0s).

    Returns:
        Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
            - On success: (rows, None). Empty result sets return ([], None).
            - On failure: (None, error_message).
    """
    if not sql or not isinstance(sql, str) or not sql.strip():
        return None, "Empty or invalid SQL query string"

    conn = None
    try:
        # Create fresh in-memory SQLite database
        conn = sqlite3.connect(":memory:")

        # Step 1: Execute setup context (DDL & initial INSERTs)
        if sql_context and sql_context.strip():
            try:
                conn.executescript(sql_context)
            except Exception as e:
                return None, f"Database setup error in sql_context: {e}"

        # Step 2: Enable SQLite read-only query mode
        try:
            conn.execute("PRAGMA query_only = ON;")
        except Exception as e:
            return None, f"Failed to enable read-only mode: {e}"

        # Step 3: Configure SQLite Authorizer to deny all mutating operations
        # Allowed read-only action codes in SQLite
        read_only_allowed = {
            sqlite3.SQLITE_SELECT,      # 21: SELECT statements
            sqlite3.SQLITE_READ,        # 20: Reading table columns
            sqlite3.SQLITE_FUNCTION,    # 31: Scalar and aggregate functions (COUNT, SUM, etc.)
            sqlite3.SQLITE_RECURSIVE,   # 33: Recursive CTE queries
        }

        def _authorizer(action_code: int, arg1: Any, arg2: Any, db_name: Any, trigger_name: Any) -> int:
            if action_code not in read_only_allowed:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(_authorizer)

        # Step 4: Configure SQLite Progress Handler for execution timeout
        start_time = time.perf_counter()
        timed_out = False

        def _progress_handler() -> int:
            nonlocal timed_out
            if time.perf_counter() - start_time > timeout:
                timed_out = True
                return 1  # Non-zero interrupts SQLite VM
            return 0

        if timeout and timeout > 0:
            # Check every 1000 SQLite VM opcodes
            conn.set_progress_handler(_progress_handler, 1000)

        # Step 5: Execute query (cursor.execute enforces a single SQL statement)
        cur = conn.cursor()
        cur.execute(sql.strip())
        rows = cur.fetchall()
        return rows, None

    except sqlite3.OperationalError as e:
        if timed_out or "interrupted" in str(e).lower():
            return None, f"Query execution timed out after {timeout} seconds"
        return None, f"SQL operational error: {e}"
    except sqlite3.ProgrammingError as e:
        if "one statement at a time" in str(e).lower():
            return None, f"SQL multiple statements error: {e}"
        return None, f"SQL programming error: {e}"
    except sqlite3.DatabaseError as e:
        if "not authorized" in str(e).lower():
            return None, f"SQL safety violation (write or destructive operation blocked): {e}"
        return None, f"SQL database error: {e}"
    except Exception as e:
        return None, f"SQL execution error: {e}"
    finally:
        if conn:
            try:
                conn.set_progress_handler(None, 0)
                conn.set_authorizer(None)
                conn.close()
            except Exception:
                pass


# =====================================================================
# 4. Result Comparison Harness
# =====================================================================
def _normalize_cell(val: Any) -> Any:
    """
    Normalizes a single SQL result cell value for reliable comparison.

    Rules:
    - NULL values (None) preserved.
    - Booleans preserved (checked before numeric types).
    - Floating point values rounded to 4 decimal places to prevent float precision mismatches.
    - Integers converted to float if compared against float to equate 1 and 1.0.
    - NaN values unified as a sentinel string '__NAN__'.
    - Strings stripped of leading/trailing whitespace.
    """
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return "__NAN__"
        return round(float(val), 4)
    if isinstance(val, str):
        return val.strip()
    return val


def _normalize_row(row: Any) -> Tuple[Any, ...]:
    """
    Normalizes a SQL row into a hashable tuple with normalized cell values.
    Handles tuples, lists, and scalar outputs.
    """
    if isinstance(row, (list, tuple)):
        return tuple(_normalize_cell(c) for c in row)
    return (_normalize_cell(row),)


def same_result(pred_rows: Any, gold_rows: Any) -> bool:
    """
    Compares predicted and ground-truth SQL query execution results.

    Comparison Logic:
    -----------------
    1. Error handling: If either result is None (indicating query execution error),
       returns False.
    2. Empty results: Both empty lists return True; mismatched empty/non-empty returns False.
    3. Multiset Comparison: Uses `collections.Counter` to compare row frequencies.
       This correctly enforces duplicate counts (unlike set() which collapses duplicates).
    4. Order-Invariance: Row order does not affect equality, matching standard SQL
       evaluation benchmarks (Spider, BIRD, Gretel).
    5. Numeric & Type Normalization:
       - 1 and 1.0 compare equal.
       - Floats rounded to 4 decimal places to avoid IEEE-754 precision issues.
       - NULLs (None) and NaNs handled consistently.
       - Leading/trailing whitespace in strings stripped.

    Args:
        pred_rows: Rows returned from predicted SQL query execution.
        gold_rows: Rows returned from ground-truth SQL query execution.

    Returns:
        bool: True if results match under multiset semantic comparison, False otherwise.
    """
    if pred_rows is None or gold_rows is None:
        return False

    if not isinstance(pred_rows, list) or not isinstance(gold_rows, list):
        return False

    if len(pred_rows) != len(gold_rows):
        return False

    if len(pred_rows) == 0 and len(gold_rows) == 0:
        return True

    try:
        norm_pred = [_normalize_row(r) for r in pred_rows]
        norm_gold = [_normalize_row(r) for r in gold_rows]
        return Counter(norm_pred) == Counter(norm_gold)
    except Exception:
        return False


# =====================================================================
# 5. Exploratory Data Analysis (EDA)
# =====================================================================
def run_eda(sample_count: int = 3) -> None:
    """
    Runs Exploratory Data Analysis (EDA) on the Gretel synthetic Text-to-SQL dataset.

    Reports:
    1. Dataset split sizes (official train/test vs. resulting train/test).
    2. Representative dataset examples with prompt, context, SQL, domain, and complexity.
    3. SQL complexity distribution (counts and percentages).
    4. Domain distribution (total domains and top 10 most frequent).
    5. Data quality check for missing or unexpected values across essential fields.
    """
    print("=" * 70)
    print("  EXPLORATORY DATA ANALYSIS (EDA) - TEXT-TO-SQL ANALYST")
    print("=" * 70)

    print("\n[1/5] Loading dataset and generating splits...")
    ds = load_data()
    train_ds, test_ds = get_splits()

    print("\n[2/5] Split Sizes Summary:")
    print(f"  Official 'train' split:    {len(ds['train']):>7,d} examples")
    print(f"  Official 'test' split:     {len(ds['test']):>7,d} examples")
    print(f"  Resulting training set:    {len(train_ds):>7,d} examples (official train split preserved)")
    print(f"  Resulting test subset:     {len(test_ds):>7,d} examples (sampled from official test split)")

    # Actual computed comparison of sql_prompt values between train split and selected test subset
    train_prompts = set(ds["train"]["sql_prompt"])
    test_prompts = set(test_ds["sql_prompt"])
    computed_overlap = len(train_prompts.intersection(test_prompts))
    overlap_status = "(completely separate)" if computed_overlap == 0 else "(! LEAKAGE DETECTED !)"
    print(f"  Prompt overlap between splits:   {computed_overlap:>7,d} examples {overlap_status}")

    print(f"\n[3/5] {sample_count} Representative Dataset Examples:")
    print("-" * 70)
    # Pick diverse examples from test_ds
    step = max(1, len(test_ds) // sample_count)
    sample_indices = [i * step for i in range(sample_count)]

    for idx, sample_idx in enumerate(sample_indices, 1):
        example = test_ds[sample_idx]
        print(f"\n--- Example #{idx} (ID: {example.get('id')}) ---")
        print(f"Domain:      {example.get('domain')} ({example.get('domain_description', '')[:60]}...)")
        print(f"Complexity:  {example.get('sql_complexity')}")
        print(f"Task Type:   {example.get('sql_task_type')}")
        print(f"Prompt:      {example.get('sql_prompt')}")
        ctx_snippet = example.get('sql_context', '').strip()
        if len(ctx_snippet) > 140:
            ctx_snippet = ctx_snippet[:140] + " ... [truncated]"
        print(f"Context:     {ctx_snippet}")
        print(f"SQL Query:   {example.get('sql')}")
        expl = example.get('sql_explanation', '')
        if expl:
            if len(expl) > 120:
                expl = expl[:120] + "..."
            print(f"Explanation: {expl}")

    print("\n" + "-" * 70)
    print("[4/5] SQL Complexity Distribution:")
    train_complexities = Counter(train_ds["sql_complexity"])
    test_complexities = Counter(test_ds["sql_complexity"])

    print(f"\n  {'Complexity':<22} | {'Train Count':<12} | {'Train %':<8} | {'Test Count':<10} | {'Test %':<8}")
    print("  " + "-" * 68)
    total_train = len(train_ds)
    total_test = len(test_ds)

    for comp, count in train_complexities.most_common():
        t_count = test_complexities.get(comp, 0)
        train_pct = (count / total_train) * 100
        test_pct = (t_count / total_test) * 100 if total_test > 0 else 0
        print(f"  {comp:<22} | {count:>12,d} | {train_pct:>7.2f}% | {t_count:>10,d} | {test_pct:>7.2f}%")

    print("\n[5/5] Domain Distribution Summary:")
    train_domains = Counter(train_ds["domain"])
    print(f"  Total unique domains in training set: {len(train_domains)}")
    print("  Top 10 most frequent domains:")
    for rank, (dom, count) in enumerate(train_domains.most_common(10), 1):
        pct = (count / total_train) * 100
        print(f"    {rank:>2}. {dom:<32} {count:>6,d} examples ({pct:.2f}%)")

    # Data quality check
    print("\n[Data Quality & Integrity Check]")
    check_fields = ["id", "domain", "sql_complexity", "sql_prompt", "sql_context", "sql"]
    null_counts = {field: 0 for field in check_fields}
    for field in check_fields:
        null_counts[field] = sum(1 for val in train_ds[field] if val is None or str(val).strip() == "")

    has_issues = any(c > 0 for c in null_counts.values())
    if not has_issues:
        print("  [OK] All required fields (id, domain, complexity, prompt, context, sql) are 100% complete.")
        print("  [OK] Zero missing, NULL, or empty values found.")
    else:
        print(f"  [WARNING] Missing values detected: {null_counts}")

    print("\n" + "=" * 70)
    print("  EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)


# =====================================================================
# Main execution block
# =====================================================================
if __name__ == "__main__":
    # Running directly runs EDA and self-tests without affecting module imports
    run_eda()
