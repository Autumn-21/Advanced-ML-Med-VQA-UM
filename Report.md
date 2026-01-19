# Final Report — Medical VQA on VQA-RAD (Baseline vs BLIP-2)

This document is a **final-report draft** aligned to the assignment rubric: background/objective, reproducible method, results with baseline comparison (including closed vs open behavior), discussion/limitations, and conclusion.

Sources of evidence:

- Baseline (prelim): [prelim/baseline_resnet50_bert.ipynb](../prelim/baseline_resnet50_bert.ipynb)
- Baseline (full-data eval): [baseline_resnet50_bert_evaluation.ipynb](../baseline_resnet50_bert_evaluation.ipynb)
- VLM (prelim): [prelim/blip2_flan_t5_xl.ipynb](../prelim/blip2_flan_t5_xl.ipynb)
- VLM (final): [blip2_flan_t5_xl_higher_final.ipynb](../blip2_flan_t5_xl_higher_final.ipynb)

Note on compute constraints: results reported here are taken from **saved notebook outputs** (no re-runs).

Appendix comparisons (more detail):

- BLIP-2 prelim vs final: [prelims_vs_fully_functional.md](prelims_vs_fully_functional.md)
- Baseline prelim vs full-data: [baseline_prelims_vs_full.md](baseline_prelims_vs_full.md)
- Results + discussion (standalone): [results_and_discussion.md](results_and_discussion.md)

---

## 1) Background (Problem Context)

Medical images (X-rays, CT, MRI) contain high-dimensional visual information that is difficult to interpret automatically. Traditional image-only models can classify abnormalities but cannot naturally answer **free-form clinical questions** about what is present in the image.

Medical Visual Question Answering (Med-VQA) addresses this by conditioning on both:

- a medical image, and
- a natural-language question,

and producing an answer (binary, short phrase, or longer free-form text). This framing tests not only recognition but also cross-modal reasoning and language grounding.

---

## 2) Objective

Build and evaluate at least two Med-VQA approaches on an open-source dataset (VQA-RAD), and critically analyze how well they handle:

- closed-ended questions (e.g., yes/no)
- open-ended questions (free-form answers)

Methods compared:

1. A **CNN+Transformer discriminative baseline**: ResNet50 (image) + BERT (question) + classifier over an answer vocabulary.
2. A **Visual-Language Model (VLM)**: BLIP-2 + Flan-T5-XL adapted with parameter-efficient fine-tuning (LoRA).

---

## 3) Method (15%) — Reproducible Experiment Setup

### 3.1 Dataset

- Dataset: Hugging Face `flaviagiammarino/vqa-rad`
- Each example contains an `image`, `question`, and `answer`.
- The repository uses the Hugging Face `train` split and constructs reproducible train/val/test splits.

### 3.2 Reproducible splitting (shared across notebooks)

Both baseline and VLM notebooks implement `load_or_create_subset_splits(...)` and share a cached indices file:

- Split cache file: `vqarad_subset_splits.json`

Procedure:

1. Load dataset and deterministically shuffle with `SEED`.
2. Select a subset of size `subset_size = max(3, int(len(ds) * SUBSET_FRAC))`.
3. Partition subset into train/val/test by `VAL_FRAC` and `TEST_FRAC` using `round(...)` (to match both notebooks).
4. Save `{config, indices}` to `vqarad_subset_splits.json`.

This ensures both methods can be evaluated on the **same examples**.

Current verified split sizes (when `SUBSET_FRAC = 1.0`, `SEED = 1337`, `VAL_FRAC = TEST_FRAC = 0.15`): **train/val/test = 1255 / 269 / 269**.

### 3.3 Preprocessing and data interfaces

Baseline preprocessing:

- Image: convert to RGB; apply ResNet50 pretrained transforms.
- Question: tokenize with `bert-base-uncased` tokenizer, pad/truncate to a fixed max length.
- Answer: normalize string (lower/whitespace), map into an integer class via a training-built vocabulary (includes `<unk>` for unseen answers).

VLM preprocessing:

- Prompt template: `"Question: {question} Answer:"`.
- Use `Blip2Processor` to construct `pixel_values` and prompt tokens.
- Tokenize the target answer; mask padding with `-100` for teacher forcing.

### 3.4 Models

#### 3.4.1 Baseline model (ResNet50 + BERT classifier)

- Image encoder: ResNet50 backbone (ImageNet weights) → 2048-d feature.
- Text encoder: BERT-base → 768-d `[CLS]` feature.
- Fusion: late fusion (prelim: concatenation; full-data: projected multiplicative fusion).
- Output: 1-of-N classification over answer vocabulary.

Key limitation by design: it cannot produce answers outside the vocabulary.

#### 3.4.2 VLM model (BLIP-2 + Flan-T5-XL with LoRA)

- Base checkpoint: `Salesforce/blip2-flan-t5-xl`.
- Vision encoder: frozen.
- Parameter-efficient adaptation: LoRA on Q-Former linear layers.

This keeps the core BLIP-2 architecture fixed while training a small number of parameters.

### 3.5 Training details (as implemented)

The notebooks make training choices explicit via top-cell constants (seed, subset fraction, epochs, batch size, LR).

Baseline training:

- Loss: cross-entropy.
- Optimizer: AdamW.
- Metrics: accuracy, weighted F1, top-5 (full-data eval notebook).

VLM training:

- Loss: BLIP-2 generation loss via teacher forcing (`labels=...`).
- Optimizer: AdamW over trainable LoRA parameters.
- Final notebook adds stability mechanisms: gradient accumulation, AMP, gradient clipping, NaN/Inf skipping, checkpointing, early stopping.

### 3.6 Environment / dependencies

Primary requirements file: [requirements.txt](../requirements.txt)

Reproducing results requires access to:

- Hugging Face dataset download
- GPU acceleration for BLIP-2 training (the final notebook was configured for an NVIDIA A10)

---

## 4) Results (25%) — Critical Analysis and Baseline Comparison

The core results and analysis are provided in:

- [results_and_discussion.md](results_and_discussion.md)

Summary of the most important comparisons (saved outputs):

- Baseline (final, split-aligned): test accuracy **0.4052**, weighted F1 **0.3844**, top-5 **0.5948** (open vs closed: **0.0887** open, **0.6759** closed).
- VLM (final): test exact match **0.4015**, ROUGE-L **0.4424**, BERTScore-F1 **0.6115**, BLEU **4.4331**.

Key takeaway: even under a shared split, the baseline and VLM are not solving identical tasks (closed-vocab classification vs free-form generation). The fairest comparison is answer-type-aware (yes/no vs open-ended) and multi-metric (exact match + semantic similarity), rather than a single headline number.

---

## 5) Discussion (15%) — Findings and Limitations

The full discussion is in:

- [results_and_discussion.md](results_and_discussion.md)

In brief:

- Metrics are brittle for short answers; BLEU is especially unstable.
- Answer-vocabulary classification is a mismatch for open-ended VQA.
- The VLM shows modest generalization and poor calibration (ECE ~0.6), which is concerning for medical settings.

---

## 6) Conclusion (2%)

This project compared a discriminative CNN+Transformer baseline against a parameter-efficient BLIP-2 VLM on VQA-RAD. Using a shared cached split (`vqarad_subset_splits.json`) enables an apples-to-apples evaluation set, but the formulations still differ (closed-vocabulary classification vs free-form generation). On the shared split, the baseline achieves **0.4052** test accuracy (weighted F1 **0.3844**, top-5 **0.5948**) and shows a large gap between inferred closed-ended performance (**0.6759**) and open-ended performance (**0.0887**), indicating that answer-vocabulary classification struggles on the long-tail. The BLIP-2 VLM, adapted via LoRA on the Q-Former, achieves **0.4015** exact match on the test split and non-trivial semantic similarity (ROUGE-L/BERTScore), suggesting partial correctness even when exact string match fails. Future work should evaluate by answer type, adopt synonym/semantic-aware scoring for open-ended answers, and improve calibration before such models can be considered reliable for clinical decision support.
