# RUNLOG — LLM 2,000-Step Speedrun

Each entry: hypothesis, changes, dev BPB before/after, conclusion.

---

## Run 0: Baseline (Unmodified Starter)

**Hypothesis**: Establish the baseline BPB with the unmodified starter code so we can measure improvement from each optimization.

**Changes**: None — running the starter code as-is.

**Config**: vocab=256 (byte), block_size=128, n_layer=4, n_head=4, n_embd=160, batch=8, lr=3e-4 (constant Adam), no weight tying, no weight decay, no grad clip, init std=0.05.

**Params**: 1,339,840

**Dev BPB**: **2.3718**

**Conclusion**: Baseline established. Many obvious problems: constant LR, no weight decay, small batch, byte-level tokenizer wastes context on Hindi, model under-uses param budget, poor init.

---

## Run 1: Quick Wins (All Training Optimizations)

**Hypothesis**: Combining weight tying, GPT-2 init (std=0.02, scaled residuals), cosine LR schedule with warmup, AdamW with weight decay, gradient clipping, and a bigger model should significantly reduce BPB.

**Changes**:
- `tie_weights = True`
- GPT-2 style init: std=0.02, residual projections scaled by 1/√(2·n_layer)
- Cosine LR schedule: warmup 100 steps, peak 1e-3, min 1e-4
- AdamW with weight_decay=0.1, betas=(0.9, 0.95)
- Gradient clipping max_norm=1.0
- Gradient accumulation: batch=16 × grad_accum=4 = effective batch 64
- Bigger model: n_layer=5, n_embd=176, n_head=8, block_size=256

**Config**: vocab=256 (byte), block_size=256, n_layer=5, n_head=8, n_embd=176, batch=16×4=64 effective

**Params**: 1,960,464

**Dev BPB**: 2.3718 → **2.2099** (−6.8%)

**Conclusion**: Significant improvement from training optimizations. Cosine LR, weight decay, and better init all contribute. Loss was still decreasing at step 2000, suggesting the model could benefit from more efficient use of steps (larger batch or better LR).

---

## Run 2: BPE Tokenizer (512 merges, vocab=768)

**Hypothesis**: A BPE tokenizer will compress Devanagari (3 bytes/char → 1-2 tokens) and common English n-grams, dramatically increasing the effective context window and improving BPB.

**Changes**:
- Trained BPE tokenizer with 512 merges on train_corpus.txt
- Vocab size: 768 (256 bytes + 512 merges)
- Compression: 7.3M bytes → 2.8M tokens (2.59x)
- Model: n_layer=4, n_embd=192, n_head=8, block_size=256 (wider, shallower to fit under 2M with larger vocab)
- batch=32×2=64 effective, max_lr=1e-3

**Params**: 1,976,448

**Dev BPB**: 2.2099 → **1.9307** (−12.6% from Run 1, −18.6% from baseline)

**Conclusion**: BPE is the single biggest improvement. The model sees ~2.6x more context per window, and the per-byte efficiency is much better. Loss curve still decreasing at step 2000 — more capacity and higher LR may help.

---

## Run 3: Deeper Model + Longer Context + Higher LR

**Hypothesis**: A deeper model (6 layers vs 4) with longer context window (512 vs 256) should capture longer-range dependencies. Higher peak LR (2e-3) should help converge faster in the 2000-step budget.

**Changes**:
- n_layer=6, n_embd=156, n_head=6, block_size=512
- max_lr=2e-3, warmup_steps=150
- batch=32×2=64 effective

**Params**: 1,964,352

**Dev BPB**: _pending_

**Conclusion**: _pending_
