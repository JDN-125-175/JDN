from load_data import Spider

spider = Spider("data/spider")
examples = spider.test

# Fine-tuned T5-small
import infer
with open("preds_finetuned.txt", "w") as f:
    for i, ex in enumerate(examples):
        if i % 10 == 0: print(f"Fine-tuned: {i}/{len(examples)}")
        f.write(infer.predict(ex["question"], ex["db_id"]) + "\n")

print("Fine-tuned predictions saved to preds_finetuned.txt")

# Pretrained T5-3B
import infer_pretrained
with open("preds_pretrained.txt", "w") as f:
    for i, ex in enumerate(examples):
        if i % 10 == 0: print(f"Pretrained: {i}/{len(examples)}")
        f.write(infer_pretrained.predict(ex["question"], ex["db_id"]) + "\n")

print("Pretrained predictions saved to preds_pretrained.txt")
