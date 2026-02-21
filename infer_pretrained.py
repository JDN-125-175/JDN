import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from load_data import Spider

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "tscholak/cxmefzzi"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

spider = Spider("data/spider")
tables = spider.load_tables(spider.tables_path)

def predict(question, db_id):
    # tscholak/cxmefzzi expects: "db_id | table1 : col1 , col2 | table2 : col3 || question"
    db = tables[db_id]
    table_names = db["table_names_original"]
    columns = db["column_names_original"]

    cols = {i: [] for i in range(len(table_names))}
    for tbl_idx, col_name in columns:
        if tbl_idx >= 0:
            cols[tbl_idx].append(col_name)

    schema_parts = [db_id]
    for i, table in enumerate(table_names):
        schema_parts.append(f"{table} : {' , '.join(cols[i])}")

    prompt = " | ".join(schema_parts) + " || " + question

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    result = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # Model outputs "db_id | sql", strip the prefix
    if " | " in result:
        result = result.split(" | ", 1)[1].strip()
    return result


if __name__ == "__main__":
    print(predict("How many singers are there?", "concert_singer"))
