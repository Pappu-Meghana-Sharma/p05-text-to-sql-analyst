"""
Data and evaluation harness 
=========================================================================
Public API:
  load_data()              - load and cache the HuggingFace dataset
  run_sql()                - safe in-memory SQLite execution with timeout
  same_result()            - order-insensitive multiset result comparison
  check_example()          - automated quality checks for one dataset row
  clean_split()            - filter a split, logging each removal with its reason
  load_excluded_ids()      - read manually flagged IDs from excluded_ids.json
  stratified_sample()      - proportional sampling across sql_complexity levels
  save_json_list()         - persist a list of IDs to a JSON file
  get_splits()             - clean-first train / val / test split generation
  self_test_harness()      - 4 assertions for run_sql and same_result
  generate_filter_report() - build and save filter_report.json
  run_eda()                - text-based EDA (the notebook provides the plots)
"""

import re, sys, json, math, time, random, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict
from itertools import zip_longest

import datasets

try:
    from config import CONFIG, SEED, QUICK_RUN
except ImportError:
    SEED = 42
    QUICK_RUN = True
    CONFIG = {
        "dataset": "gretelai/synthetic_text_to_sql",
        "test_size": 200,
        "quick_test_size": 20,
        "val_size": 200,
        "test_ids_path": "test_ids.json",
        "val_ids_path": "val_ids.json",
        "train_ids_path": "train_ids.json",
        "excluded_ids_path": "excluded_ids.json",
        "filter_report_path": "filter_report.json",
    }

_DATASET_CACHE: Optional[datasets.DatasetDict] = None

# Detects write or DDL keywords at the start of a SQL string
_WRITE_RE = re.compile(
    r"^\s*(UPDATE|DELETE|INSERT|DROP|ALTER|CREATE|REPLACE|TRUNCATE|MERGE)\b",
    re.IGNORECASE,
)


# -- 1. Dataset Loading --------------------------------------------------------

def load_data() -> datasets.DatasetDict:
    """Loads and memory-caches the configured HuggingFace dataset."""
    global _DATASET_CACHE
    if _DATASET_CACHE is not None:
        return _DATASET_CACHE

    name = CONFIG.get("dataset", "gretelai/synthetic_text_to_sql")
    try:
        ds = datasets.load_dataset(name)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load '{name}': {e}\n"
            "Check internet access, disk space, or set HF_TOKEN."
        ) from e

    if not {"train", "test"}.issubset(ds.keys()):
        raise RuntimeError(f"Expected 'train' and 'test' splits, got: {list(ds.keys())}")

    required = {"id", "domain", "sql_complexity", "sql_prompt", "sql_context", "sql"}
    for split in ("train", "test"):
        missing = required - set(ds[split].column_names)
        if missing:
            raise RuntimeError(f"Split '{split}' missing columns: {missing}")

    _DATASET_CACHE = ds
    return _DATASET_CACHE


# -- 2. SQL Execution Sandbox --------------------------------------------------

def run_sql(
    sql_context: Optional[str],
    sql: str,
    timeout: float = 5.0,
) -> Tuple[Optional[List[Tuple]], Optional[str]]:
    """
    Runs sql in a fresh in-memory SQLite database seeded from sql_context.

    Safety layers:
      - PRAGMA query_only blocks writes at the SQLite level.
      - set_authorizer denies any non-read action (INSERT, UPDATE, DELETE, DROP, ...).
      - Progress handler kills queries that exceed timeout seconds.
      - cursor.execute() rejects multiple statements in one call.

    Returns (rows, None) on success, (None, error_message) on any failure.
    """
    if not sql or not isinstance(sql, str) or not sql.strip():
        return None, "Empty or invalid SQL string"

    conn = None
    timed_out = False  # initialized before try so the except block can always read it
    try:
        conn = sqlite3.connect(":memory:")

        if sql_context and sql_context.strip():
            try:
                conn.executescript(sql_context)
            except Exception as e:
                return None, f"Context setup error: {e}"

        try:
            conn.execute("PRAGMA query_only = ON;")
        except Exception as e:
            return None, f"Could not enable read-only mode: {e}"

        _allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }

        def _auth(action, *_):
            return sqlite3.SQLITE_OK if action in _allowed else sqlite3.SQLITE_DENY

        conn.set_authorizer(_auth)

        start = time.perf_counter()

        def _progress():
            nonlocal timed_out
            if time.perf_counter() - start > timeout:
                timed_out = True
                return 1
            return 0

        if timeout > 0:
            conn.set_progress_handler(_progress, 1000)

        cur = conn.cursor()
        cur.execute(sql.strip())
        return list(cur.fetchall()), None

    except sqlite3.OperationalError as e:
        if timed_out or "interrupted" in str(e).lower():
            return None, f"Query timed out after {timeout}s"
        return None, f"SQL operational error: {e}"
    except sqlite3.ProgrammingError as e:
        if "one statement" in str(e).lower():
            return None, f"Multiple statements not allowed: {e}"
        return None, f"SQL programming error: {e}"
    except sqlite3.DatabaseError as e:
        if "not authorized" in str(e).lower():
            return None, f"Write/DDL operation blocked: {e}"
        return None, f"SQL database error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"
    finally:
        if conn:
            try:
                conn.set_progress_handler(None, 0)
                conn.set_authorizer(None)
                conn.close()
            except Exception:
                pass


# -- 3. Result Comparison ------------------------------------------------------

def _norm_cell(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return "__NAN__" if (isinstance(val, float) and math.isnan(val)) else round(float(val), 4)
    if isinstance(val, str):
        return val.strip()
    return val


def _norm_row(row: Any) -> Tuple:
    if isinstance(row, (list, tuple)):
        return tuple(_norm_cell(c) for c in row)
    return (_norm_cell(row),)


def same_result(pred_rows: Any, gold_rows: Any) -> bool:
    """
    Order-insensitive multiset comparison of SQL result rows.
    Handles numeric tolerance (1 == 1.0), NULL, NaN, and string whitespace.
    Returns False if either argument is None (signals an execution error).
    """
    if pred_rows is None or gold_rows is None:
        return False
    if not isinstance(pred_rows, list) or not isinstance(gold_rows, list):
        return False
    if len(pred_rows) != len(gold_rows):
        return False
    if not pred_rows:
        return True
    try:
        return Counter(_norm_row(r) for r in pred_rows) == Counter(_norm_row(r) for r in gold_rows)
    except Exception:
        return False


# -- 4. Data Cleaning ----------------------------------------------------------

def check_example(row: Dict) -> Tuple[bool, Optional[str]]:
    """
    Automated quality gate for one dataset row.
    Returns (True, None) if clean, or (False, reason) if it should be removed.

    Checks applied in order:
      1. SQL must start with SELECT or WITH - no writes or DDL allowed.
      2. SQL must execute without error against sql_context.
      3. Gold result must not be empty or trivially matchable (single zero or None).
    """
    sql = (row.get("sql") or "").strip()
    ctx = row.get("sql_context") or ""

    if _WRITE_RE.match(sql):
        op = sql.split()[0].upper() if sql.split() else "?"
        return False, f"Non-read operation - query starts with {op}"

    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return False, f"Does not start with SELECT or WITH: '{sql[:60]}'"

    rows, err = run_sql(ctx, sql)
    if err is not None:
        return False, f"SQL execution failed: {err}"

    if not rows:
        return False, "Gold query returns empty result ([])"

    # Single-cell zero or None - a wrong query can match this by luck
    if len(rows) == 1 and len(rows[0]) == 1:
        val = rows[0][0]
        if val is None:
            return False, f"Gold query returns trivial result: {rows}"
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val == 0:
            return False, f"Gold query returns trivial result: {rows}"

    return True, None


def load_excluded_ids(path: Path) -> Set[int]:
    """Returns the set of manually flagged IDs from excluded_ids.json."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids: Set[int] = set()
        for key in ("test_exclusions", "train_exclusions"):
            for entry in data.get(key, []):
                ids.add(int(entry["id"]))
        return ids
    except FileNotFoundError:
        return set()
    except Exception as e:
        print(f"[Warning] Could not load {path}: {e}", file=sys.stderr)
        return set()


def clean_split(
    ds_split: datasets.Dataset,
    split_name: str,
    excluded_ids: Optional[Set[int]] = None,
    show_progress: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filters ds_split and returns (clean_rows, removal_log).

    Each removal_log entry: { id, split, reason, sql_prompt[:100], sql[:100] }

    Removals are applied in order:
      1. IDs listed in excluded_ids (manual audit flags from excluded_ids.json).
      2. Non-read / DDL operations (UPDATE, DELETE, INSERT, DROP, ...).
      3. SQL execution errors (syntax, missing table, unsupported dialect, ...).
      4. Empty gold result ([]).
      5. Trivial single-cell result ([(0,)] / [(None,)]) - matchable by chance.
    """
    excluded_ids = excluded_ids or set()
    clean: List[Dict] = []
    log: List[Dict] = []
    total = len(ds_split)

    for i, row in enumerate(ds_split):
        rid = row.get("id")

        if rid in excluded_ids:
            log.append({
                "id": rid, "split": split_name,
                "reason": "Manual exclusion (see excluded_ids.json)",
                "sql_prompt": (row.get("sql_prompt") or "")[:100],
                "sql": (row.get("sql") or "")[:100],
            })
        else:
            valid, reason = check_example(row)
            if valid:
                clean.append(dict(row))
            else:
                log.append({
                    "id": rid, "split": split_name,
                    "reason": reason,
                    "sql_prompt": (row.get("sql_prompt") or "")[:100],
                    "sql": (row.get("sql") or "")[:100],
                })

        if show_progress and (i + 1) % 1000 == 0:
            print(
                f"\r  {split_name}: {i+1:,}/{total:,} | "
                f"clean: {len(clean):,} | removed: {len(log):,}   ",
                end="", flush=True,
            )

    if show_progress:
        print(
            f"\r  {split_name}: {total:,}/{total:,} | "
            f"clean: {len(clean):,} | removed: {len(log):,}"
        )

    return clean, log


# -- 5. Split Helpers ----------------------------------------------------------

def save_json_list(path: Path, ids: List[int]) -> None:
    """Saves a list of integer IDs to a JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
    except Exception as e:
        print(f"[Warning] Could not write {path}: {e}", file=sys.stderr)


def _load_json_list(path: Path) -> List[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [int(x) for x in raw] if isinstance(raw, list) else []
    except Exception:
        return []


def stratified_sample(
    rows: List[Dict],
    n: int,
    rng: random.Random,
    key: str = "sql_complexity",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Samples exactly n rows from rows, proportional to each value of key.

    The sampled list is interleaved across groups so the first K items
    span all complexity levels - useful for the QUICK_RUN subset.
    Returns (sampled, remainder).
    """
    if n >= len(rows):
        return rows.copy(), []

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        groups[row.get(key, "unknown")].append(row)

    total = len(rows)
    alloc: Dict[str, int] = {k: max(1, round(n * len(g) / total)) for k, g in groups.items()}

    # Adjust total allocation to exactly n
    diff = n - sum(alloc.values())
    for k in sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True):
        if diff == 0:
            break
        if diff > 0:
            alloc[k] = min(alloc[k] + 1, len(groups[k]))
            diff -= 1
        elif diff < 0 and alloc[k] > 1:
            alloc[k] -= 1
            diff += 1

    per_group_sampled: List[List[Dict]] = []
    remainder: List[Dict] = []
    for k, g in groups.items():
        shuffled = g.copy()
        rng.shuffle(shuffled)
        per_group_sampled.append(shuffled[: alloc[k]])
        remainder.extend(shuffled[alloc[k]:])

    # Interleave groups so sampled[:K] spans all complexity levels
    sampled = [
        item
        for items in zip_longest(*per_group_sampled)
        for item in items
        if item is not None
    ]
    return sampled, remainder


# -- 6. Clean-First Split Generation -------------------------------------------

def get_splits(
    seed: int = SEED,
    quick_run: Optional[bool] = None,
    save_ids: bool = True,
) -> Tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Returns (train_ds, val_ds, test_ds) using a clean-first pipeline.

    Pipeline:
      1. If all three ID files exist, load them (fast path).
         Otherwise, run full cleaning on both official splits and save IDs.
      2. test_ds  : 200 stratified clean examples from the official test split.
                    First 20 IDs are the QUICK_RUN subset.
                    QUICK_RUN=True returns only those 20.
      3. val_ds   : 200 stratified clean examples from the remaining test pool,
                    disjoint from test. Used for prompt / hyperparameter selection.
      4. train_ds : All clean examples from the official train split (~52k).
                    Used for RAG indexing and fine-tuning.
    """
    ds = load_data()
    if quick_run is None:
        quick_run = CONFIG.get("QUICK_RUN", QUICK_RUN)

    test_size  = CONFIG.get("test_size", 200)
    val_size   = CONFIG.get("val_size", 200)
    quick_size = CONFIG.get("quick_test_size", 20)

    test_ids_path  = Path(CONFIG.get("test_ids_path",      "test_ids.json"))
    val_ids_path   = Path(CONFIG.get("val_ids_path",        "val_ids.json"))
    train_ids_path = Path(CONFIG.get("train_ids_path",      "train_ids.json"))
    excluded_path  = Path(CONFIG.get("excluded_ids_path",   "excluded_ids.json"))

    # Fast path: all ID files already generated by the notebook
    if test_ids_path.exists() and val_ids_path.exists() and train_ids_path.exists():
        test_ids  = _load_json_list(test_ids_path)
        val_ids   = _load_json_list(val_ids_path)
        train_ids = _load_json_list(train_ids_path)
    else:
        # Full cleaning pipeline (runs once; saves IDs so subsequent calls are instant)
        excluded = load_excluded_ids(excluded_path)
        rng = random.Random(seed)

        print("Cleaning official test split ...")
        clean_test, _ = clean_split(ds["test"],  "test",  excluded, show_progress=True)
        print("Cleaning official train split ...")
        clean_train, _ = clean_split(ds["train"], "train", excluded, show_progress=True)

        test_rows, rest = stratified_sample(clean_test, test_size, rng)
        val_rows, _ = stratified_sample(rest, val_size, rng)

        test_ids = [r["id"] for r in test_rows]
        val_ids = [r["id"] for r in val_rows]
        train_ids = [r["id"] for r in clean_train]

        if save_ids:
            save_json_list(test_ids_path, test_ids)
            save_json_list(val_ids_path, val_ids)
            save_json_list(train_ids_path, train_ids)
            print(f"Saved: {test_ids_path}, {val_ids_path}, {train_ids_path}")

    def _select(official_split: datasets.Dataset, ids: List[int]) -> datasets.Dataset:
        id2idx = {tid: i for i, tid in enumerate(official_split["id"])}
        indices = [id2idx[tid] for tid in ids if tid in id2idx]
        return official_split.select(indices)

    train_ds = _select(ds["train"], train_ids)
    val_ds   = _select(ds["test"],  val_ids)

    active_ids = test_ids[:quick_size] if quick_run else test_ids
    test_ds    = _select(ds["test"], active_ids)

    return train_ds, val_ds, test_ds


# -- 7. Self-Test Harness ------------------------------------------------------

def self_test_harness() -> None:
    """
    Four assertions that verify run_sql and same_result behave correctly.
    Raises AssertionError with a descriptive message if any test fails.
    """
    ctx = "CREATE TABLE t (x INT); INSERT INTO t VALUES (1), (2);"

    # 1. Gold vs gold must always match
    rows, err = run_sql(ctx, "SELECT x FROM t;")
    assert err is None,          f"Test 1 failed - unexpected error: {err}"
    assert same_result(rows, rows), "Test 1 failed - gold vs gold should be True"

    # 2. DDL write operations must be blocked
    _, err = run_sql(ctx, "DROP TABLE t;")
    assert err is not None, "Test 2 failed - DROP TABLE should have been blocked"

    # 3. A query referencing a missing table must return an error
    _, err = run_sql(ctx, "SELECT * FROM table_does_not_exist_xyz;")
    assert err is not None, "Test 3 failed - bad table reference should return an error"

    # 4. A runaway query must time out
    infinite = (
        "WITH RECURSIVE r(c) AS (VALUES(0) UNION ALL SELECT c+1 FROM r) "
        "SELECT * FROM r;"
    )
    _, err = run_sql(ctx, infinite, timeout=0.1)
    assert err is not None and "timed out" in err.lower(), (
        f"Test 4 failed - infinite query should time out, got: {err}"
    )

    print("All 4 self-tests passed [OK]")


# -- 8. Filter Report ----------------------------------------------------------

def _categorize_reason(reason: str) -> str:
    r = reason.lower()
    if "manual" in r:           return "manual_exclusion"
    if "non-read" in r or "does not start" in r: return "write_or_non_select"
    if "context setup" in r:    return "context_setup_error"
    if "sql execution failed" in r: return "sql_execution_error"
    if "empty result" in r:     return "empty_result"
    if "trivial" in r:          return "trivial_result"
    return "other"


def generate_filter_report(
    test_log: List[Dict],
    train_log: List[Dict],
    test_ids: List[int],
    val_ids: List[int],
    train_ids: List[int],
    test_raw: int = 5851,
    train_raw: int = 100000,
    manual_audit: Optional[Dict] = None,
) -> Dict:
    """
    Builds and saves filter_report.json. Returns the report dict.
    Called from the notebook after cleaning both splits.
    """
    def _counts(log: List[Dict]) -> Dict[str, int]:
        return dict(Counter(_categorize_reason(e["reason"]) for e in log))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_split": {
            "raw_total":         test_raw,
            "removed_total":     len(test_log),
            "removed_by_reason": _counts(test_log),
            "clean_pool_size":   test_raw - len(test_log),
        },
        "train_split": {
            "raw_total":         train_raw,
            "removed_total":     len(train_log),
            "removed_by_reason": _counts(train_log),
            "clean_total":       len(train_ids),
        },
        "final_splits": {
            "test":             len(test_ids),
            "val":              len(val_ids),
            "train_clean_total": len(train_ids),
            "quick_run_subset": CONFIG.get("quick_test_size", 20),
        },
        "manual_audit": manual_audit or {
            "status": "pending manual review (1 example pre-flagged as reference)",
            "test_sampled":        1,
            "test_flagged":        1,
            "test_flag_rate_pct":  round(1 / 60 * 100, 2),
            "train_sampled":       0,
            "train_flagged":       0,
            "train_flag_rate_pct": 0.0,
            "note": "ID 4323 (test): BRICS query returns all rows, no country filter applied.",
        },
    }

    report_path = Path(CONFIG.get("filter_report_path", "filter_report.json"))
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Filter report saved -> {report_path}")
    except Exception as e:
        print(f"[Warning] Could not save filter report: {e}", file=sys.stderr)

    return report


# -- 9. EDA (text-based) -------------------------------------------------------

def run_eda(sample_count: int = 3) -> None:
    """
    Text-based EDA summary. The notebook provides the corresponding plots.
    """
    print("=" * 70)
    print("  EDA - TEXT-TO-SQL ANALYST")
    print("=" * 70)

    ds = load_data()
    train_ds, val_ds, test_ds = get_splits()

    print(f"\n  Official train split : {len(ds['train']):>8,d} rows")
    print(f"  Official test split  : {len(ds['test']):>8,d} rows")
    print(f"  Clean train_ds       : {len(train_ds):>8,d} rows")
    print(f"  Val set (val_ds)     : {len(val_ds):>8,d} rows")
    print(f"  Test set (test_ds)   : {len(test_ds):>8,d} rows")
    print(f"  QUICK_RUN uses first {CONFIG.get('quick_test_size', 20)} test IDs")

    print(f"\n  {sample_count} representative test examples:")
    print("  " + "-" * 66)
    step = max(1, len(test_ds) // sample_count)
    for i in range(sample_count):
        ex = test_ds[i * step]
        print(f"\n  ID {ex['id']} | {ex['domain']} | {ex['sql_complexity']}")
        print(f"  Prompt : {ex['sql_prompt']}")
        print(f"  SQL    : {ex['sql']}")

    print("\n\n  SQL Complexity Distribution:")
    train_ctr = Counter(train_ds["sql_complexity"])
    val_ctr   = Counter(val_ds["sql_complexity"])
    test_ctr  = Counter(test_ds["sql_complexity"])
    print(f"  {'Complexity':<25} {'Train':>9} {'Val':>6} {'Test':>6}")
    print("  " + "-" * 52)
    for comp, cnt in train_ctr.most_common():
        print(
            f"  {comp:<25} {cnt:>9,d} "
            f"{val_ctr.get(comp, 0):>6,d} {test_ctr.get(comp, 0):>6,d}"
        )

    print("\n  Domain Distribution - Top 10:")
    for rank, (dom, cnt) in enumerate(Counter(train_ds["domain"]).most_common(10), 1):
        print(f"  {rank:>2}. {dom:<35} {cnt:>7,d}")

    print("\n" + "=" * 70)


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    self_test_harness()
    run_eda()