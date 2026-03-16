Text-to-SQL with T5 and PICARD-style Constrained Decoding
==========================================================
CS175 Project - Team JDN

Members: Daniel Meng, Jackson Yan, Novyanna Tsang

Fine-tuned T5-small on the Spider text-to-SQL benchmark, with a custom
PICARD-inspired constrained decoding strategy that validates SQL structure
during beam search to reduce invalid outputs.


Libraries / Packages Used
--------------------------
- PyTorch (torch)
- HuggingFace Transformers (transformers)
- sqlite3 (Python standard library)


Online / Public Repository Code Used or Adapted
-------------------------------------------------
- Spider dataset: https://yale-lily.github.io/spider
- T5 model (t5-small): https://huggingface.co/t5-small
- PICARD pretrained model (tscholak/cxmefzzi): https://huggingface.co/tscholak/cxmefzzi
- PICARD paper (Scholak et al., 2021): https://arxiv.org/abs/2109.05093
  Our constrained decoding logic in infer_picard.py is inspired by the PICARD
  approach but is a simplified custom implementation, not the original code.


Code We Wrote
--------------
All files in src/:

  load_data.py           - Loads and preprocesses Spider dataset; builds prompts
  train.py               - Fine-tunes T5-small on Spider training set
  infer.py               - Inference using the fine-tuned T5 model
  infer_picard.py        - Inference with PICARD-style constrained decoding
  infer_pretrained.py    - Inference using a pretrained PICARD model (baseline)
  evaluate.py            - Execution-based evaluation (accuracy, Jaccard, difficulty breakdown)
  generate_predictions.py - Batch prediction generation for evaluation
