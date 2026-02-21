import re
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict
from collections import Counter

def get_db_path(data_path: str, db_id: str, db_subdir: str = "database") -> Path:
    """Path to the SQLite file for a Spider database (e.g. data/spider/database/concert_singer/concert_singer.sqlite)."""
    base = Path(data_path) / db_subdir / db_id
    return base / f"{db_id}.sqlite"


def execute_sql(db_path: Path, sql: str) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
    """
    Run a SQL string on the given SQLite database.
    Returns (rows, error_message) where:
    - rows: list of rows (each row is a tuple), or None if execution fails
    - error_message: error string if execution fails, None if successful
    """
    if not db_path.exists():
        return None, f"Database file not found: {db_path}"
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return [tuple(r) for r in rows], None
    except sqlite3.OperationalError as e:
        return None, f"OperationalError: {str(e)}"
    except sqlite3.ProgrammingError as e:
        return None, f"ProgrammingError: {str(e)}"
    except sqlite3.IntegrityError as e:
        return None, f"IntegrityError: {str(e)}"
    except sqlite3.DatabaseError as e:
        return None, f"DatabaseError: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def _format_result_for_display(rows: Optional[List[Tuple[Any, ...]]], max_rows: int = 5, max_cell_len: int = 40) -> str:
    """Format query result for debug printing: first max_rows rows, truncate long cells."""
    if rows is None:
        return "ERROR (execution failed)"
    if not rows:
        return "[]  (empty)"
    lines = []
    for i, row in enumerate(rows[:max_rows]):
        cells = []
        for c in row:
            s = str(c)
            if len(s) > max_cell_len:
                s = s[: max_cell_len - 3] + "..."
            cells.append(s)
        lines.append("  " + str(tuple(cells)))
    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def _normalize_cell(x: Any) -> Any:
    """Normalize one cell so that 1 and 1.0 compare equal, and strings are trimmed."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return int(x) if float(x) == int(x) else float(x)
        except (ValueError, OverflowError):
            return float(x) if isinstance(x, float) else int(x)
    if isinstance(x, str):
        return x.strip()
    return x


def normalize_results(rows: Optional[List[Tuple[Any, ...]]]) -> Optional[Tuple[Tuple[Any, ...], ...]]:
    """
    Normalize and sort query results so we can compare two result sets.
    Returns a sorted tuple of normalized rows, or None if rows is None.
    """
    if rows is None:
        return None
    normalized = [tuple(_normalize_cell(c) for c in row) for row in rows]
    return tuple(sorted(normalized, key=lambda row: str(row)))


def jaccard_similarity(
    gold_rows: Optional[List[Tuple[Any, ...]]],
    pred_rows: Optional[List[Tuple[Any, ...]]],
) -> Optional[float]:
    """
    Calculate Jaccard similarity between two result sets.
    Returns similarity score (0.0 to 1.0), or None if either execution failed.
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    """
    if gold_rows is None or pred_rows is None:
        return None
    
    gold_set = set(normalize_results(gold_rows) or ())
    pred_set = set(normalize_results(pred_rows) or ())
    
    if not gold_set and not pred_set:
        return 1.0  # Both empty, perfect match
    
    intersection = len(gold_set & pred_set)
    union = len(gold_set | pred_set)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def execution_match(
    gold_rows: Optional[List[Tuple[Any, ...]]],
    pred_rows: Optional[List[Tuple[Any, ...]]],
) -> bool:
    """
    True if predicted query results match gold (same set of rows).
    Order of rows does not matter. Returns False if either execution failed.
    """
    g = normalize_results(gold_rows)
    p = normalize_results(pred_rows)
    if g is None:
        return False
    return g == p


def get_difficulty(sql: str) -> str:
    """
    Categorize SQL difficulty based on its components, approximating the
    official Spider difficulty rating (easy / medium / hard / extra hard).
    """
    s = sql.upper()

    num_selects   = s.count('SELECT')
    has_nested    = num_selects > 1
    has_set_op    = bool(re.search(r'\b(UNION|INTERSECT|EXCEPT)\b', s))
    has_groupby   = 'GROUP BY' in s
    has_having    = 'HAVING' in s
    has_orderby   = 'ORDER BY' in s
    has_limit     = 'LIMIT' in s
    num_joins     = len(re.findall(r'\bJOIN\b', s))

    where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)', s, re.DOTALL)
    num_conditions = 0
    if where_match:
        clause = where_match.group(1)
        num_conditions = len(re.findall(r'\b(AND|OR)\b', clause)) + 1

    num_components = sum([
        'WHERE' in s,
        has_groupby,
        has_orderby,
        has_limit,
        has_having,
        has_set_op,
    ])

    if has_set_op or num_selects >= 3:
        return 'extra hard'
    elif has_nested or has_having or num_components >= 3 or num_conditions >= 4:
        return 'hard'
    elif has_groupby or num_joins >= 1 or num_conditions >= 2 or num_components >= 2:
        return 'medium'
    else:
        return 'easy'


def evaluate_execution(
    data_path: str,
    examples: List[dict],
    predictions: List[str],
    db_subdir: str = "database",
) -> Tuple[float, List[dict]]:
    """
    For each example: run gold SQL and predicted SQL on the right database,
    then compare result sets. Returns (accuracy, list of per-example results).

    examples: list of dicts with "query" (gold SQL) and "db_id".
    predictions: list of predicted SQL strings, same length and order as examples.
    """
    if len(examples) != len(predictions):
        raise ValueError(f"examples length ({len(examples)}) != predictions length ({len(predictions)})")

    results = []
    correct_count = 0

    for ex, pred_sql in zip(examples, predictions):
        db_id = ex["db_id"]
        gold_sql = ex["query"]
        db_path = get_db_path(data_path, db_id, db_subdir)

        # Run both queries
        gold_rows, gold_error = execute_sql(db_path, gold_sql)
        pred_rows, pred_error = execute_sql(db_path, pred_sql)

        gold_ok = gold_rows is not None
        pred_ok = pred_rows is not None
        
        # Determine error type
        if not gold_ok:
            err = "gold_sql_failed"
            error_msg = gold_error
        elif not pred_ok:
            err = "pred_sql_failed"
            error_msg = pred_error
        else:
            err = None
            error_msg = None

        # Calculate match and similarity
        match = execution_match(gold_rows, pred_rows)
        similarity = jaccard_similarity(gold_rows, pred_rows) if gold_ok and pred_ok else None
        
        if match:
            correct_count += 1

        results.append({
            "correct": match,
            "gold_ok": gold_ok,
            "pred_ok": pred_ok,
            "error": err,
            "error_message": error_msg,
            "similarity": similarity,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "gold_row_count": len(gold_rows) if gold_rows else 0,
            "pred_row_count": len(pred_rows) if pred_rows else 0,
        })

    accuracy = correct_count / len(examples) if examples else 0.0
    return accuracy, results


def run_evaluation(
    data_path: str = "data/spider",
    split: str = "dev",
    predictions_file: Optional[str] = None,
    model_ckpt: Optional[str] = "t5_spider_ckpt",
    max_examples: Optional[int] = 100,
    db_subdir: str = "database",
    debug_n: int = 0,
) -> float:
    """
    Load Spider dev (or train) examples, get predicted SQL (from file or from model),
    then compute execution accuracy. Default: 100 examples.

    If debug_n > 0, print gold SQL, pred SQL, and execution results for the first debug_n examples.

    Returns the accuracy (0.0 to 1.0).
    """
    from load_data import Spider

    spider = Spider(data_path)
    tables = spider.load_tables(spider.tables_path)

    if split == "dev":
        examples = spider.test
    else:
        examples = spider.train

    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]

    if predictions_file is not None:
        with open(predictions_file, "r") as f:
            predictions = [line.strip() for line in f if line.strip()]
    else:
        from infer import predict as model_predict
        predictions = []
        for i, ex in enumerate(examples):
            if i % 50 == 0:
                print(f"Predicting example {i}/{len(examples)}")
            predictions.append(model_predict(ex["question"], ex["db_id"]))


    acc, results = evaluate_execution(data_path, examples, predictions, db_subdir)

    n = len(examples)
    correct = sum(1 for r in results if r["correct"])
    pred_fail = sum(1 for r in results if not r["pred_ok"])
    gold_fail = sum(1 for r in results if not r["gold_ok"])
    
    # Calculate percentages
    pred_fail_pct = (pred_fail / n * 100) if n > 0 else 0.0
    gold_fail_pct = (gold_fail / n * 100) if n > 0 else 0.0
    
    # Calculate similarity statistics for successful queries
    similarities = [r["similarity"] for r in results if r["similarity"] is not None]
    avg_similarity = sum(similarities) / len(similarities) if similarities else None
    
    # Analyze error types
    pred_errors = [r["error_message"] for r in results if not r["pred_ok"] and r["error_message"]]
    gold_errors = [r["error_message"] for r in results if not r["gold_ok"] and r["error_message"]]
    
    # Categorize errors
    def categorize_error(error_msg: str) -> str:
        """Categorize error message into a type."""
        error_lower = error_msg.lower()
        if "no such table" in error_lower or "no such column" in error_lower:
            return "Schema Error"
        elif "syntax error" in error_lower or "near" in error_lower:
            return "Syntax Error"
        elif "operationalerror" in error_lower:
            return "Operational Error"
        elif "programmingerror" in error_lower:
            return "Programming Error"
        elif "database file not found" in error_lower:
            return "Database Not Found"
        else:
            return "Other Error"
    
    pred_error_types = Counter([categorize_error(e) for e in pred_errors])
    gold_error_types = Counter([categorize_error(e) for e in gold_errors])

    print(f"\n{'='*80}")
    print(f"Execution accuracy: {acc:.4f} ({correct}/{n})")
    print(f"{'='*80}")
    
    # Error statistics
    print(f"\nError Statistics:")
    print(f"  Pred SQL execution errors: {pred_fail}/{n} ({pred_fail_pct:.2f}%)")
    if gold_fail:
        print(f"  Gold SQL execution errors: {gold_fail}/{n} ({gold_fail_pct:.2f}%) (check DBs)")
    
    # Similarity statistics
    if similarities:
        print(f"\nSimilarity Statistics (for queries that executed successfully):")
        print(f"  Average Jaccard similarity: {avg_similarity:.4f}")
        print(f"  Perfect matches (similarity=1.0): {sum(1 for s in similarities if s == 1.0)}/{len(similarities)}")
        print(f"  High similarity (>=0.8): {sum(1 for s in similarities if s >= 0.8)}/{len(similarities)}")
        print(f"  Medium similarity (0.5-0.8): {sum(1 for s in similarities if 0.5 <= s < 0.8)}/{len(similarities)}")
        print(f"  Low similarity (<0.5): {sum(1 for s in similarities if s < 0.5)}/{len(similarities)}")
    
    # Error breakdown
    if pred_error_types:
        print(f"\nPred SQL Error Breakdown:")
        for error_type, count in pred_error_types.most_common():
            print(f"  {error_type}: {count} ({count/pred_fail*100:.1f}% of pred errors)")
    
    if gold_error_types:
        print(f"\nGold SQL Error Breakdown:")
        for error_type, count in gold_error_types.most_common():
            print(f"  {error_type}: {count} ({count/gold_fail*100:.1f}% of gold errors)")

    # Difficulty breakdown
    difficulty_order = ['easy', 'medium', 'hard', 'extra hard']
    difficulty_results = {d: {'correct': 0, 'total': 0} for d in difficulty_order}
    for r in results:
        d = get_difficulty(r['gold_sql'])
        difficulty_results[d]['total'] += 1
        if r['correct']:
            difficulty_results[d]['correct'] += 1

    print(f"\nAccuracy by Difficulty:")
    for d in difficulty_order:
        dr = difficulty_results[d]
        if dr['total'] > 0:
            pct = dr['correct'] / dr['total'] * 100
            print(f"  {d.capitalize():12s}: {dr['correct']:4d}/{dr['total']:4d} ({pct:.1f}%)")
    
    # Show sample errors
    if pred_errors:
        print(f"\nSample Pred SQL Errors (showing up to 5):")
        for i, error_msg in enumerate(pred_errors[:5]):
            # Find corresponding example
            example_idx = next(j for j, r in enumerate(results) if not r["pred_ok"] and r["error_message"] == error_msg)
            ex = examples[example_idx]
            print(f"  [{i+1}] {categorize_error(error_msg)}")
            print(f"      Error: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}")
            print(f"      DB: {ex['db_id']}")
            print(f"      SQL: {predictions[example_idx][:80]}{'...' if len(predictions[example_idx]) > 80 else ''}")
    
    print(f"\n{'='*80}\n")

    if debug_n > 0:
        n_show = min(debug_n, len(examples))
        print("\n" + "=" * 60 + " DEBUG (first {} examples) ".format(n_show) + "=" * 60)
        for i in range(n_show):
            ex = examples[i]
            pred_sql = predictions[i]
            gold_sql = ex["query"]
            db_path = get_db_path(data_path, ex["db_id"], db_subdir)
            gold_rows, _ = execute_sql(db_path, gold_sql)
            pred_rows, _ = execute_sql(db_path, pred_sql)
            r = results[i]
            print("\n--- Example {} (db_id={}) ---".format(i + 1, ex["db_id"]))
            print("Question: {}".format(ex.get("question", "")[:80] + ("..." if len(ex.get("question", "")) > 80 else "")))
            print("Gold SQL:  {}".format(gold_sql))
            print("Pred SQL: {}".format(pred_sql))
            print("Gold result:\n{}".format(_format_result_for_display(gold_rows)))
            print("Pred result:\n{}".format(_format_result_for_display(pred_rows)))
            print("Match: {}  (gold_ok={}, pred_ok={})".format(r["correct"], r["gold_ok"], r["pred_ok"]))
            if r.get("similarity") is not None:
                print("Jaccard similarity: {:.4f}".format(r["similarity"]))
            if r.get("error_message"):
                print("Error: {}".format(r["error_message"]))
        print("=" * 100 + "\n")

    return acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Execution accuracy for Spider text-to-SQL")
    parser.add_argument("--data_path", default="data/spider", help="Path to Spider data")
    parser.add_argument("--split", default="dev", choices=("dev", "train"), help="dev or train")
    parser.add_argument("--pred", dest="predictions_file", default=None, help="File with one predicted SQL per line (optional)")
    parser.add_argument("--model", dest="model_ckpt", default="t5_spider_ckpt", help="Model checkpoint if not using --pred")
    parser.add_argument("--max", type=int, default=100, help="Max examples (default: 100; use 0 for all)")
    parser.add_argument("--db_subdir", default="database", help="Subdir under data_path with db_id folders")
    parser.add_argument("--debug", type=int, default=0, metavar="N", help="Print gold/pred SQL and results for first N examples (e.g. 5)")
    args = parser.parse_args()

    run_evaluation(
        data_path=args.data_path,
        split=args.split,
        predictions_file=args.predictions_file,
        model_ckpt=args.model_ckpt,
        max_examples=args.max,
        db_subdir=args.db_subdir,
        debug_n=args.debug,
    )
