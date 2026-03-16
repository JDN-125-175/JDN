import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from load_data import Spider, process_query

class SpiderDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_len=512, max_target_len=256):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src = self.inputs[idx]
        tgt = self.targets[idx]

        src_enc = self.tokenizer(
            src, truncation=True, padding="max_length",
            max_length=self.max_input_len, return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt, truncation=True, padding="max_length",
            max_length=self.max_target_len, return_tensors="pt"
        )

        input_ids = src_enc["input_ids"].squeeze(0)
        attention_mask = src_enc["attention_mask"].squeeze(0)
        labels = tgt_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spider = Spider("data/spider")
    tables = spider.load_tables(spider.tables_path)
    train_inputs, train_targets = process_query(spider.train, tables)

    # train_inputs, train_targets = train_inputs[:2000], train_targets[:2000]

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    ds = SpiderDataset(train_inputs, train_targets, tokenizer)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)  # I'd lower from 3e-4

    num_epochs = 7

    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0.0

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            global_step += 1

            if step % 50 == 0:
                print(f"epoch {epoch+1}/{num_epochs} step {step} loss {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"epoch {epoch+1} done | avg_loss={avg_loss:.4f}")

    model.save_pretrained("t5_spider_ckpt")
    tokenizer.save_pretrained("t5_spider_ckpt")
    print("saved -> t5_spider_ckpt/")

if __name__ == "__main__":
    main()
