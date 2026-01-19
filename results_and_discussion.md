# Results (25%) — Critical Analysis and Baseline Comparison

This section critically analyzes the **saved outputs** from:

- Baseline (discriminative): [baseline_resnet50_bert_evaluation.ipynb](../baseline_resnet50_bert_evaluation.ipynb)
- Our VLM (generative): [blip2_flan_t5_xl_higher_final.ipynb](../blip2_flan_t5_xl_higher_final.ipynb)

Numbers below reflect the latest notebook outputs currently saved in this repo.

---

## 2.1 What is being compared (and why this is tricky)

The baseline and the VLM solve *related but not identical* problems:

- **Baseline (ResNet50 + BERT)** is a **train-derived answer-vocabulary classifier** (size depends on the train split; currently 357 on the shared split).
  - It cannot output answers outside that label set.
  - It is naturally advantaged on frequent short answers (especially yes/no) but struggles with rare/open-ended answers.

- **BLIP-2 + Flan-T5-XL (LoRA on Q-Former)** is a **free-form generator**.
  - It can output answers not seen verbatim.
  - Exact-match metrics can be harsh (synonyms/phrasing count as wrong), but semantic metrics (ROUGE/BERTScore) can capture partial correctness.

Because of this mismatch, the most honest comparison is:

- Compare **closed-ended behavior** (yes/no) separately from open-ended.
- Use **multiple metrics** rather than a single headline number.

Fairness note: earlier iterations of this repo used slightly different cached split logic (different cache file + `int(...)` vs `round(...)` sizing), which meant the baseline and VLM were not guaranteed to be computed on identical test examples.

Update: the baseline notebook now uses the same cached split file and rounding logic as the VLM (`vqarad_subset_splits.json`). With this change, the split-aligned rerun produces `train/val/test = 1255/269/269`, enabling a fair test-set comparison.

---

## 2.2 Headline results (saved outputs)

### Baseline (full-data evaluation)

From [baseline_resnet50_bert_evaluation.ipynb](../baseline_resnet50_bert_evaluation.ipynb):

- Split sizes (shared protocol): train/val/test = **1255 / 269 / 269**
- Answer vocabulary size (train-derived): **357**

Final evaluation (split-aligned rerun):

- Test Accuracy: **0.4052**
- Weighted F1: **0.3844**
- Top-5 Accuracy: **0.5948**
- Open vs Closed (inferred):
  - Closed-ended (Yes/No) Accuracy: **0.6759** (145 samples)
  - Open-ended Accuracy: **0.0887** (124 samples)

### Our VLM (BLIP-2 “final”)

From [blip2_flan_t5_xl_higher_final.ipynb](../blip2_flan_t5_xl_higher_final.ipynb):

- Dataset split sizes: train/val/test = **1255 / 269 / 269**
- Model selection: best checkpoint by validation BLEU

Final evaluation (best checkpoint):

- Validation:
  - Exact match / “accuracy”: **0.4424**
  - BLEU: **7.2407**
  - ROUGE-L: **0.4828**
  - BERTScore-F1: **0.6700**
  - ECE: **0.5576**
- Test:
  - Exact match / “accuracy”: **0.4015**
  - BLEU: **4.4331**
  - ROUGE-L: **0.4424**
  - BERTScore-F1: **0.6115**
  - ECE: **0.5985**

Operational metrics (test):

- Mean latency: **0.4263 s/sample**
- Throughput: **~9.28 samples/s**

---

## 2.3 Critical comparison: which is “better”?

### If you only look at top-1 accuracy/exact match

- Baseline test accuracy: **0.4052**
- VLM test exact match: **0.4015**

With the split-alignment in place, this is now an apples-to-apples comparison on the same test indices.

However, this is not the full story because the baseline’s score is dominated by label-frequency effects and the fact that many questions map to common short answers.

### The key finding: baseline collapses on open-ended questions
The baseline collapses on open-ended questions under a flat answer-vocabulary classifier framing (long-tail labels + paraphrase fragmentation). On the shared split rerun, the baseline reaches **0.6759** accuracy on inferred closed-ended questions (yes/no) but only **0.0887** on open-ended questions.
That is a very strong signal that:

- the label space is too long-tailed for a flat classifier,
- synonym/paraphrase variation becomes separate labels,
- many open-ended answers are effectively “unlearnable” as single labels with limited data.

In other words, the baseline is mostly a **closed-ended (yes/no) solver** plus a weak long-tail classifier.

### The VLM is weaker on exact-match BLEU, but stronger on “semantic closeness”
The VLM’s BLEU is low (test BLEU **4.43**), but ROUGE-L and BERTScore are non-trivial (test ROUGE-L **0.4424**, BERTScore-F1 **0.6115**).

Interpretation:

- VQA-RAD answers are often very short.
- BLEU becomes unstable/uninformative for short strings.
- BERTScore/ROUGE suggest the model is often *in the right semantic neighborhood* even when exact string match fails.

So if you care about **clinical plausibility** rather than exact string identity, the VLM results are more promising than BLEU alone implies.

### Closed-ended performance: baseline still looks strong
The baseline generally performs much better on inferred yes/no questions than on open-ended answers.
The VLM’s reported “closed-ended” precision/recall/F1 in the notebook is implemented as a **binary ‘yes’ detector** across all questions, which is not perfectly aligned with a true “yes/no subset” evaluation.
So, we should not claim the VLM beats the baseline on yes/no based on that reporting.

---

## 2.4 Interesting findings

1) **Task framing matters more than architecture tweaks**
   - Changing concat vs multiplicative fusion in the baseline is secondary compared to the fact that classification forces open-ended answers into a long-tail label space.

2) **Baseline is “good” mostly where the dataset is easiest**
   - High yes/no accuracy indicates the dataset contains many questions answerable by coarse cues and common priors.

3) **VLM generalization gap (val → test) is noticeable**
   - VLM exact match drops from **0.4424 (val)** to **0.4015 (test)**.
   - This suggests limited robustness and/or mild overfitting despite LoRA.

4) **Calibration is poor for the VLM (ECE ~0.6)**
   - Even when accuracy is moderate, the model’s confidence (as approximated here) is misaligned with correctness.
   - For medical use, calibration matters; this is a red flag.

5) **Evaluation metrics disagree (especially for short answers)**
   - Low BLEU with decent exact match + semantic scores indicates metric brittleness.
   - This supports moving toward answer-type-aware evaluation and/or human review.

---

# Discussion (15%) — Findings and Limitations

## 3.1 Findings (what we can responsibly claim)

- A flat answer-vocabulary classifier is strongly biased toward closed-ended answers (especially yes/no), and struggles on open-ended answers due to long-tail label sparsity and paraphrase fragmentation.
- A generative VLM with LoRA-on-Q-Former achieves ~0.40 exact match on the test split, and shows non-trivial semantic similarity scores (ROUGE-L/BERTScore), suggesting partial correctness even when exact match fails.

## 3.2 Limitations (methodological)

- **Single-source split**: both approaches split the Hugging Face “train” split into train/val/test. This is convenient but not an official benchmark split and may not reflect true external generalization.
- **Brittle evaluation**:
  - Exact match penalizes valid synonyms (“left pleura” vs “left hemithorax”).
  - BLEU is unreliable for short answers.
  - Baseline accuracy does not credit partially correct open-ended responses.
- **Open/closed inference is heuristic**: baseline’s open/closed analysis infers type from the answer text, which is imperfect.

## 3.3 Limitations (modeling)

- **Baseline formulation mismatch**: ~357-way classification (train-derived) is not a natural representation of open-ended VQA.
- **VLM adaptation is limited**: LoRA only on Q-Former may be insufficient for medical-domain language grounding; it adapts cross-modal alignment but may not fully adapt the language model to medical phrasing.
- **Prompt and truncation sensitivity**:
  - The VLM uses fixed max lengths (question/answer), which can silently truncate information.
- **Calibration not addressed**: VLM ECE around 0.6 suggests it is not reliable for risk-sensitive decisions.

## 3.4 Practical constraints

- Compute constraints restrict extensive hyperparameter search, ablations (e.g., LoRA rank, target modules), and multi-seed reporting.
- Larger models, longer training, and richer evaluations (e.g., radiologist review) are desirable but expensive.

## 3.5 What we would do next (highest ROI)

- **Evaluate by answer type** (yes/no vs free-form) with separate metrics and decoding constraints.
- **Add synonym-aware scoring** for anatomy/location terms and common medical paraphrases.
- **Human evaluation** on a curated subset to assess clinical correctness.
- **Improve VLM calibration** (temperature scaling on logits for yes/no subset; uncertainty estimation).
- Consider **structured prediction** for open-ended answers (organ/modality/finding slots) as a middle ground between classification and free-form generation.
