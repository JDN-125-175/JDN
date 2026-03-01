import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from load_data import Spider, process_tables

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("t5_spider_ckpt")
model = T5ForConditionalGeneration.from_pretrained("t5_spider_ckpt").to(device)
model.eval()

spider = Spider("data/spider")
tables = spider.load_tables(spider.tables_path)

def predict(question, db_id):

    schema = process_tables(tables[db_id])

    prompt = (
        "Task: Text-to-SQL. "
        f"Question: {question} "
        f"Database Schema: {schema} "
        "SQL Query:"
    )


    enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=128,        
            num_beams=8,               
            length_penalty=0.8,        
            no_repeat_ngram_size=3,    
            early_stopping=True,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # TEST
    print(predict("How many singers are there?", "concert_singer"))
