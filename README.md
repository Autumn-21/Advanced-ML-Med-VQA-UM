# Medical Visual Question Answering on VQA-RAD: Discriminative Baseline vs. BLIP-2 VLM

A comparative study of two approaches to Medical Visual Question Answering (Med-VQA): a **CNN+Transformer discriminative baseline** (ResNet50 + BERT) and a **parameter-efficient fine-tuned Vision-Language Model** (BLIP-2 + Flan-T5-XL with LoRA), evaluated on the [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) dataset. Built as an Advanced Machine Learning project by students of the Master of Artificial Intelligence programme at Universiti Malaya.

## Team

| Name | Programme | Institution |
|---|---|---|
| Raden Ahmad Faiz | Master of Artificial Intelligence | Universiti Malaya |
| Pyae Sone Khant Aung (Junaid) | Master of Artificial Intelligence | Universiti Malaya |

## Problem

Radiology images (X-ray, CT, MRI) carry information that's hard to query automatically. Medical VQA conditions a model on both an image and a free-text clinical question, and asks it to produce an answer — testing recognition, cross-modal reasoning, and language grounding, not just classification. This project builds and critically compares two ways of approaching that: treating it as closed-vocabulary classification versus open-ended generation.

## Approaches Compared

| | Baseline | VLM |
|---|---|---|
| **Architecture** | ResNet50 (image) + BERT-base (question), fused, classified over an answer vocabulary | BLIP-2 (frozen vision encoder) + Flan-T5-XL, adapted with **LoRA on the Q-Former** |
| **Task framing** | 1-of-N classification (357-way, train-derived vocabulary) | Free-form text generation |
| **Can answer outside training vocab?** | No | Yes |
| **Training** | Cross-entropy loss, AdamW, cosine LR schedule, 15 epochs | Teacher-forced generation loss, AdamW on LoRA params, gradient accumulation + AMP + early stopping, 20 epochs |

## Dataset & Reproducible Splits

- Source: Hugging Face [`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) (image, question, answer triples).
- Both notebooks call a shared `load_or_create_subset_splits(...)` function that deterministically shuffles (`SEED = 1337`) and splits the data, caching the resulting indices to `vqarad_subset_splits.json` — this guarantees the baseline and the VLM are evaluated on **exactly the same test examples**, which is what makes the comparison meaningful.
- Split sizes (full dataset, `SUBSET_FRAC = 1.0`): **train / val / test = 1255 / 269 / 269**.

## Results

Numbers are from saved notebook outputs (see [`results_and_discussion.md`](results_and_discussion.md) for full detail).

**Baseline (ResNet50 + BERT), test set:**

| Metric | Score |
|---|---|
| Accuracy | 0.4052 |
| Weighted F1 | 0.3844 |
| Top-5 Accuracy | 0.5948 |
| Closed-ended (yes/no) accuracy | 0.6759 |
| Open-ended accuracy | 0.0887 |

**VLM (BLIP-2 + Flan-T5-XL, LoRA), test set:**

| Metric | Score |
|---|---|
| Exact match | 0.4015 |
| BLEU | 4.4331 |
| ROUGE-L | 0.4424 |
| BERTScore-F1 | 0.6115 |
| ECE (calibration error) | 0.5985 |
| Mean latency | 0.4263 s/sample |

### Key finding

Top-line accuracy looks similar (0.4052 vs 0.4015), but the two models are not solving the same problem. The baseline is essentially **a strong yes/no solver** (0.6759 accuracy) that **collapses on open-ended questions** (0.0887) — a symptom of forcing free-form answers into a long-tailed, paraphrase-fragmented label space. The VLM's exact-match score is comparable, but its low BLEU alongside decent ROUGE-L/BERTScore suggests it's often semantically close even when it misses the exact string — and its poor calibration (ECE ≈ 0.6) is a concern for any clinical-adjacent use. Full discussion of these takeaways, along with methodological limitations, is in [`results_and_discussion.md`](results_and_discussion.md).

## Repository Structure

```
├── Report.md                                # Full write-up: background, method, results, discussion, conclusion
├── results_and_discussion.md                # Standalone results + critical analysis + limitations
├── baseline_resnet50_bert_evaluation.ipynb   # Baseline model: training + full evaluation
├── blip2_flan_t5_xl_higher_final.ipynb       # VLM: LoRA fine-tuning + evaluation
├── vqarad_subset_splits.json                 # Cached, shared train/val/test indices (for reproducibility)
└── test_predictions_for_annotation.csv       # Sample predictions vs. references, for qualitative/human review
```

> **Note:** `Report.md` also links to a `prelim/` folder and two appendix files (`baseline_prelims_vs_full.md`, `prelims_vs_fully_functional.md`) plus a `requirements.txt`, which weren't present in this archive — make sure those are included when you push, or update the links in `Report.md` if they're no longer part of the repo.

## Running It

Both notebooks expect GPU acceleration (the VLM notebook was configured for an NVIDIA A10) and will download the dataset from Hugging Face on first run.

```bash
pip install torch torchvision transformers datasets scikit-learn peft sacrebleu rouge-score bert-score tqdm matplotlib
```

1. Run `baseline_resnet50_bert_evaluation.ipynb` to train/evaluate the discriminative baseline. It will create `vqarad_subset_splits.json` if it doesn't already exist.
2. Run `blip2_flan_t5_xl_higher_final.ipynb` to fine-tune and evaluate the VLM — it reuses the same cached split file, so results are directly comparable to the baseline.

## Limitations

- Splits come from a single source (the HF `train` split partitioned in-house) rather than an official train/test benchmark split.
- Exact-match evaluation penalizes valid synonyms/paraphrases; BLEU is unstable on the very short answers common in VQA-RAD.
- Open vs. closed question type is inferred heuristically from answer text, not from ground-truth metadata.
- LoRA adaptation was applied only to the Q-Former, which may be insufficient for full medical-domain language grounding.
- Compute constraints limited hyperparameter search, ablations, and multi-seed reporting.

See [`Report.md`](Report.md) for the full method write-up and [`results_and_discussion.md`](results_and_discussion.md) for the complete results, discussion, and proposed next steps (answer-type-aware evaluation, synonym-aware scoring, calibration, human evaluation).
