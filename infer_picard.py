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


class PicardLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            ids = [t for t in input_ids[b].tolist() if t not in SKIP_IDS]
            if not ids:
                continue
            sql = "".join(tokenizer.convert_ids_to_tokens(ids) or []).replace("▁", " ").strip()
            if sql and not sql.startswith("select"):
                scores[b, :] = -float("inf")
                scores[b, tokenizer.eos_token_id] = 0.0
        return scores


def predict(question, db_id):
    schema = process_tables(tables[db_id])
    prompt = build_prompt(question.lower(), schema)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            logits_processor=LogitsProcessorList([PicardLogitsProcessor()]),
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

