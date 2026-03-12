import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, LogitsProcessor
from load_data import Spider, process_tables, build_prompt
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("t5_spider_ckpt")
model = T5ForConditionalGeneration.from_pretrained("t5_spider_ckpt").to(device)
model.eval()

spider = Spider("data/spider")
tables = spider.load_tables(spider.tables_path)


class PicardLogitsProcessor(LogitsProcessor):
    """
    PICARD-style constraints for T5 decoding.
    Prunes impossible tokens according to SQL syntax + schema.
    """

    def __init__(self, schema_text: str):
        self.schema_text = schema_text.lower()
        self.allowed_tokens = self.get_allowed_tokens(schema_text)
        self.table_columns = self.extract_columns(schema_text)
        self.keywords = [
            "select", "from", "where", "and", "or", "group", "by", "order",
            "limit", "as", "count", "sum", "avg", "min", "max", "*", "desc",
            "asc", "having", "join", "on", "union", "intersect", "except"
        ]

    def get_allowed_tokens(self, schema_text: str):
        tokens = set()
        for piece in schema_text.replace("_", " ").replace(".", " ").split():
            ids = tokenizer(piece, add_special_tokens=False).input_ids
            tokens.update(ids)
        for kw in self.keywords:
            ids = tokenizer(kw, add_special_tokens=False).input_ids
            tokens.update(ids)
        tokens.update([tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id])
        return tokens

    def extract_columns(self, schema_text: str):
        """
        Return set of table.column strings present in schema
        """
        cols = set()
        for line in schema_text.splitlines():
            if line.strip().startswith("-"):
                col = line.strip()[2:]  # remove "- "
                cols.add(col)
        return cols

    def prune_tokens(self, decoded_str: str):
        """
        Determine valid tokens given the decoded SQL so far
        """
        # allowed keywords + schema columns
        valid_tokens = set(self.allowed_tokens)

        # FROM -> only table names
        if re.search(r"\bfrom\s+$", decoded_str):
            # Only allow table names after FROM
            table_names = {t.split()[1].lower() for t in re.findall(r"table:\s*(\S+)", self.schema_text)}
            table_tokens = set()
            for t in table_names:
                table_tokens.update(tokenizer(t, add_special_tokens=False).input_ids)
            valid_tokens = table_tokens | {tokenizer.eos_token_id}

        # JOIN columns 
        if re.search(r"\bjoin\s+", decoded_str):
            join_cols = self.table_columns
            col_tokens = set()
            for col in join_cols:
                for piece in col.split("."):
                    col_tokens.update(tokenizer(piece, add_special_tokens=False).input_ids)
            valid_tokens = col_tokens | {tokenizer.eos_token_id}

        return valid_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # decode last few tokens instead of full sequence
        last_tokens = input_ids[:, -5:]
        decoded = tokenizer.batch_decode(last_tokens, skip_special_tokens=True)[0].lower()

        valid_tokens = self.prune_tokens(decoded)

        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(valid_tokens)] = 0

        return scores + mask


def predict(question, db_id, max_len=256, num_beams=4):
    schema = process_tables(tables[db_id])
    prompt = build_prompt(question.lower(), schema)
    processor = PicardLogitsProcessor(schema)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True,
            logits_processor=[processor],
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(predict("How many singers are there?", "concert_singer"))