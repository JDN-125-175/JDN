import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from load_data import Spider, process_query


spider = Spider("data/spider")
tables = spider.load_tables(spider.tables_path)
inputs, results = process_query(spider.train, tables)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

