"""Optimized trainer with cosine LR schedule, warmup, AdamW, gradient
clipping, and gradient accumulation.

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    # cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # separate weight decay for non-bias, non-layernorm params
    decay_params = []
    no_decay_params = []
    for pn, p in model.named_parameters():
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    opt_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(opt_groups, lr=args.max_lr, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        # set LR for this step
        lr = get_lr(step, args.warmup_steps, args.steps, args.max_lr, args.min_lr)
        for pg in opt.param_groups:
            pg['lr'] = lr

        # gradient accumulation
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            _, loss = model(x, y)
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            accum_loss += loss.item()
        accum_loss /= args.grad_accum

        # gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()
        losses.append(accum_loss)
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.6f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    # every public config attribute is saved — if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
