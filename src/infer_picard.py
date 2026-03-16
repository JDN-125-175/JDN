import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, LogitsProcessor, LogitsProcessorList
from load_data import Spider, build_prompt, process_tables

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5_spider_ckpt")
model = T5ForConditionalGeneration.from_pretrained("t5_spider_ckpt").to(device)
model.eval()

spider = Spider("data/spider")
tables = spider.load_tables(spider.tables_path)

SKIP_IDS = {tokenizer.pad_token_id, tokenizer.eos_token_id, model.config.decoder_start_token_id} - {None}

SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "limit",
    "join", "left", "right", "inner", "outer", "on", "as", "distinct", "and",
    "or", "not", "in", "between", "like", "is", "null", "count", "sum", "avg",
    "min", "max", "asc", "desc", "all", "exists", "union", "intersect", "except",
    "case", "when", "then", "else", "end", "cast",
}

ALIAS_RE = re.compile(r"^(t\d+|[a-z])$")
SINGLE_CLAUSES = ["select", "from", "where", "group by", "having", "order by", "limit"]


def get_schema(db_id):
    db = tables[db_id]
    table_names = {t.lower() for t in db["table_names_original"]}
    col_names   = {col.lower() for tbl_idx, col in db["column_names_original"] if tbl_idx >= 0}
    col_names.add("*")
    return table_names, col_names


def token_is_word_start(token_str):
    return token_str.startswith("▁") or token_str in ("(", ")", ",", ";", "=", "<", ">")


def is_valid_sql_prefix(sql, table_names, col_names):
    if not sql:
        return True
    if not sql.startswith("select"):
        return False

    all_ids = table_names | col_names

    # clause ordering
    outer, depth = [], 0
    for ch in sql:
        if ch == "(": depth += 1
        elif ch == ")": depth = max(0, depth - 1)
        outer.append(" " if depth > 0 else ch)
    outer_sql = "".join(outer)

    last_pos = -1
    for clause in SINGLE_CLAUSES:
        positions = [m.start() for m in re.finditer(rf"\b{re.escape(clause)}\b", outer_sql)]
        if len(positions) > 1:
            return False
        if positions and positions[0] < last_pos:
            return False
        if positions:
            last_pos = positions[0]

    # FROM/JOIN table check
    in_str = False
    quote_char = None
    for match in re.finditer(r"\b(?:from|join)\s+([\w.]+)", sql, re.IGNORECASE):
        pos = match.start()
        # recompute string context up to this position
        in_str, quote_char = False, None
        for ch in sql[:pos]:
            if not in_str and ch in ('"', "'"):
                in_str, quote_char = True, ch
            elif in_str and ch == quote_char:
                in_str = False
        if in_str:
            continue
        word = match.group(1).lower().split(".")[0]
        if ALIAS_RE.match(word) or word in SQL_KEYWORDS:
            continue
        if word not in table_names:
            return False

    return True


class PicardLogitsProcessor(LogitsProcessor):
    def __init__(self, table_names, col_names):
        self.table_names = table_names
        self.col_names   = col_names

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            ids = [t for t in input_ids[b].tolist() if t not in SKIP_IDS]
            if not ids:
                continue

            # only validate at word boundaries
            last_token = tokenizer.convert_ids_to_tokens(ids[-1])
            if last_token and not token_is_word_start(last_token):
                continue  # mid-word token: don't check yet, let the word finish

            sql = "".join(
                tokenizer.convert_ids_to_tokens(i) or "" for i in ids
            ).replace("▁", " ").strip()

            if not is_valid_sql_prefix(sql, self.table_names, self.col_names):
                scores[b, :] = -float("inf")
                scores[b, tokenizer.eos_token_id] = 0.0

        return scores


def predict(question, db_id):
    schema = process_tables(tables[db_id])
    prompt = build_prompt(question.lower(), schema)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    table_names, col_names = get_schema(db_id)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            logits_processor=LogitsProcessorList([
                PicardLogitsProcessor(table_names, col_names)
            ]),
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
