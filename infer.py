import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from load_data import Spider, process_tables

def predict(question, db_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained("t5_spider_ckpt")
    model = T5ForConditionalGeneration.from_pretrained("t5_spider_ckpt").to(device)
    model.eval()

    spider = Spider("data/spider")
    tables = spider.load_tables(spider.tables_path)
    schema = process_tables(tables[db_id])

    prompt = (
        "translate to SQL:\n"
        f"question: {question}\n"
        f"table: {schema}\n"
    )

    enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_length=256,
            num_beams=4,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # TEST
    print(predict("How many singers are there?", "concert_singer"))
