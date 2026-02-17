import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple

def get_db_path(data_path: str, db_id: str, db_subdir: str = "database") -> Path:
    """Path to the SQLite file for a Spider database (e.g. data/spider/database/concert_singer/concert_singer.sqlite)."""
    base = Path(data_path) / db_subdir / db_id
    return base / f"{db_id}.sqlite"


def execute_sql(db_path: Path, sql: str) -> Optional[List[Tuple[Any, ...]]]:
    """
    Run a SQL string on the given SQLite database.
    Returns a list of rows (each row is a tuple), or None if execution fails
    (syntax error, missing file, etc.).
    """
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return [tuple(r) for r in rows]
    except (sqlite3.Error, Exception):
        return None


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
        gold_rows = execute_sql(db_path, gold_sql)
        pred_rows = execute_sql(db_path, pred_sql)

        gold_ok = gold_rows is not None
        pred_ok = pred_rows is not None
        if not gold_ok:
            err = "gold_sql_failed"
        elif not pred_ok:
            err = "pred_sql_failed"
        else:
            err = None

        match = execution_match(gold_rows, pred_rows)
        if match:
            correct_count += 1

        results.append({
            "correct": match,
            "gold_ok": gold_ok,
            "pred_ok": pred_ok,
            "error": err,
            "db_id": db_id,
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

    print(f"Execution accuracy: {acc:.4f} ({correct}/{n})")
    if pred_fail:
        print(f"  Pred SQL execution errors: {pred_fail}")
    if gold_fail:
        print(f"  Gold SQL execution errors: {gold_fail} (check DBs)")

    if debug_n > 0:
        n_show = min(debug_n, len(examples))
        print("\n" + "=" * 60 + " DEBUG (first {} examples) ".format(n_show) + "=" * 60)
        for i in range(n_show):
            ex = examples[i]
            pred_sql = predictions[i]
            gold_sql = ex["query"]
            db_path = get_db_path(data_path, ex["db_id"], db_subdir)
            gold_rows = execute_sql(db_path, gold_sql)
            pred_rows = execute_sql(db_path, pred_sql)
            r = results[i]
            print("\n--- Example {} (db_id={}) ---".format(i + 1, ex["db_id"]))
            print("Question: {}".format(ex.get("question", "")[:80] + ("..." if len(ex.get("question", "")) > 80 else "")))
            print("Gold SQL:  {}".format(gold_sql))
            print("Pred SQL: {}".format(pred_sql))
            print("Gold result:\n{}".format(_format_result_for_display(gold_rows)))
            print("Pred result:\n{}".format(_format_result_for_display(pred_rows)))
            print("Match: {}  (gold_ok={}, pred_ok={})".format(r["correct"], r["gold_ok"], r["pred_ok"]))
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
