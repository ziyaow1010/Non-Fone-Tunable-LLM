import os
import random
import argparse
import tempfile
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
    get_cosine_schedule_with_warmup,
)

@dataclass
class PMPArgs:
    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    mode: str = "pmp"  

    rho: float = 0.50
    tau: int = 200  

    eta: float = 2e-5
    eta_u: float = 2e-5

    lam: float = 3.0
    alpha: float = 0.0

    R: float = 2.0
    clip_ascent_norm: float = 3.0
    clip_ticket_norm: float = 1.0

    unlearn_accum: int = 8

    t_EB: int = 500
    EB_iou_thresh: float = 0.99
    EB_patience: int = 5

    batch_size: int = 4
    grad_accum: int = 8
    warmup_steps: int = 2000
    max_len: int = 256
    train_steps: int = 20000
    seed: int = 42

    dataset_name: str = "DKYoon/SlimPajama-6B"
    dataset_config: str = "default"
    text_field: str = "text"
    streaming: bool = True
    shuffle_buffer: int = 50_000

    out_dir: str = "/root/autodl-tmp/result0.5"

    log_unlearn_every: int = 1
    save_every: int = 0  

    log_every: int = 200
    ema_beta: float = 0.98


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load_state_dict(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_global_exact_topk_mask_from_grads(model, rho: float):
    flats = []
    metas = []
    offset = 0

    for name, p in model.named_parameters():
        n = p.numel()
        if p.grad is None:
            g = torch.zeros(n, device=p.device, dtype=torch.float32)
        else:
            g = p.grad.detach().abs().float().reshape(-1)
        flats.append(g)
        metas.append((name, offset, offset + n, p.shape))
        offset += n

    scores = torch.cat(flats, dim=0)
    k = max(1, int(rho * scores.numel()))
    topk_idx = torch.topk(scores, k, largest=True).indices

    flat_mask = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    flat_mask[topk_idx] = True
    mask = {name: flat_mask[s:e].view(shape) for name, s, e, shape in metas}
    return mask


def mask_iou(mask_a, mask_b):
    inter = 0
    uni = 0
    for k in mask_a.keys():
        if k not in mask_b:
            continue
        a, b = mask_a[k], mask_b[k]
        inter += torch.logical_and(a, b).sum().item()
        uni += torch.logical_or(a, b).sum().item()
    return inter / (uni + 1e-12)


def save_mask_packed_atomic(M, path: str):
    packed, meta = {}, {}
    for name, m in M.items():
        m_cpu = m.detach().to("cpu").contiguous().view(-1).numpy().astype(np.uint8)
        packed[name] = torch.from_numpy(np.packbits(m_cpu))
        meta[name] = {"shape": list(m.shape), "numel": int(m.numel())}
    obj = {"packed": packed, "meta": meta}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="mask_", suffix=".pt", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@torch.no_grad()
def project_complement_l2_ball_(model, M, R: float):
    sqsum = 0.0
    for name, p in model.named_parameters():
        if name in M:
            comp = p.data[~M[name]]
            sqsum += (comp.float() ** 2).sum().item()
    norm = sqsum ** 0.5

    if norm > R and norm > 0:
        scale = R / (norm + 1e-12)
        for name, p in model.named_parameters():
            if name in M:
                p.data[~M[name]] *= scale
    return min(norm, R)


def zero_complement_grads_(model, M):
    for name, p in model.named_parameters():
        if p.grad is not None and name in M:
            p.grad.data[~M[name]] = 0


def complement_grad_norm(model, M):
    vecs = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name not in M:
            continue
        vecs.append(p.grad.detach()[~M[name]].float().flatten())
    return torch.norm(torch.cat(vecs), p=2).item() if vecs else 0.0


def initialize_model(args: PMPArgs, device: str):
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
    )
    return LlamaForCausalLM(config).to(device)


class PackedTextBlocks(IterableDataset):
    def __init__(self, hf_iterable, tokenizer, max_len: int, text_field: str, eos_id: int):
        super().__init__()
        self.ds = hf_iterable
        self.tok = tokenizer
        self.max_len = max_len
        self.text_field = text_field
        self.eos_id = eos_id

    def __iter__(self):
        buf = []
        for ex in self.ds:
            try:
                text = ex.get(self.text_field)
                if not text:
                    continue
                ids = self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]
                buf.extend(ids + [self.eos_id])

                while len(buf) >= self.max_len:
                    block = buf[:self.max_len]
                    buf = buf[self.max_len:]
                    input_ids = torch.tensor(block, dtype=torch.long)
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids),
                        "labels": input_ids.clone(),
                    }
            except Exception:
                continue


def collate_fixed_blocks(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def _load_stream_dataset(args: PMPArgs, split="train", seed=None):
    kwargs = {"split": split, "streaming": args.streaming}
    ds = (
        load_dataset(args.dataset_name, args.dataset_config, **kwargs)
        if args.dataset_config != "default"
        else load_dataset(args.dataset_name, **kwargs)
    )
    if args.streaming:
        ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=seed if seed is not None else args.seed)
    return ds


def make_stream_loaders(tokenizer, args: PMPArgs):
    pre_ds = PackedTextBlocks(
        _load_stream_dataset(args, seed=args.seed + 7),
        tokenizer, args.max_len, args.text_field, tokenizer.eos_token_id
    )
    train_ds = PackedTextBlocks(
        _load_stream_dataset(args, seed=args.seed),
        tokenizer, args.max_len, args.text_field, tokenizer.eos_token_id
    )

    pre_loader = DataLoader(
        pre_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fixed_blocks,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fixed_blocks,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return pre_loader, train_loader


def next_batch_safe(it_obj, loader, max_skip: int = 1000):
    skipped = 0
    while True:
        try:
            return next(it_obj), it_obj
        except StopIteration:
            it_obj = iter(loader)
        except Exception:
            skipped += 1
            if skipped >= max_skip:
                raise RuntimeError("Too many consecutive batch failures in streaming iterator.")
            continue


def earlybird_discovery(model, pre_loader, args: PMPArgs, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.eta, weight_decay=0.0)

    last_mask = None
    stable = 0
    it = iter(pre_loader)

    for _t in range(args.t_EB):
        batch, it = next_batch_safe(it, pre_loader)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        cur_mask = build_global_exact_topk_mask_from_grads(model, rho=args.rho)

        if last_mask is not None:
            iou = mask_iou(cur_mask, last_mask)
            stable = stable + 1 if iou >= args.EB_iou_thresh else 0
            if stable >= args.EB_patience:
                return cur_mask

        last_mask = cur_mask
        opt.step()

    return last_mask


def base_train(model, train_loader, args: PMPArgs, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.eta, weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.train_steps)

    it = iter(train_loader)

    for t in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            B, it = next_batch_safe(it, train_loader)
            B = {k: v.to(device, non_blocking=True) for k, v in B.items()}
            loss = model(**B).loss / args.grad_accum
            loss.backward()
            accum_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (t + 1) % args.log_every == 0:
            print(f"[base t={t}] loss={accum_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if args.save_every > 0 and (t + 1) % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": t + 1,
                "args": asdict(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"BASE_ckpt_step{t+1}.pt"))

    return model


def pmp_train_stage2(model, train_loader, M, args: PMPArgs, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.eta, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.train_steps)

    it_train = iter(train_loader)
    it_unl = iter(train_loader)
    unlearn_count = 0

    ema = None  

    for t in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(args.grad_accum):
            B, it_train = next_batch_safe(it_train, train_loader)
            B = {k: v.to(device, non_blocking=True) for k, v in B.items()}
            loss = model(**B).loss / args.grad_accum
            loss.backward()
            total_loss += float(loss.item())

        zero_complement_grads_(model, M)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_ticket_norm)
        optimizer.step()
        scheduler.step()

        if ema is None:
            ema = total_loss
        else:
            beta = args.ema_beta
            ema = beta * ema + (1 - beta) * total_loss

        if t > 0 and (t % args.tau == 0):
            model.zero_grad(set_to_none=True)

            loss_u_avg = 0.0
            for _ in range(args.unlearn_accum):
                Bu, it_unl = next_batch_safe(it_unl, train_loader)
                Bu = {k: v.to(device, non_blocking=True) for k, v in Bu.items()}
                lu = model(**Bu).loss / args.unlearn_accum
                lu.backward()
                loss_u_avg += float(lu.item())

            gnorm = complement_grad_norm(model, M)
            clip_scale = min(1.0, args.clip_ascent_norm / (gnorm + 1e-12))

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if name not in M:
                        continue
                    if args.alpha > 0:
                        p.data[M[name]] -= args.eta_u * args.alpha * p.grad.data[M[name]]

            comp_norm_after = project_complement_l2_ball_(model, M, args.R)
            unlearn_count += 1

            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

            if unlearn_count % args.log_unlearn_every == 0:
                print(
                    f"[unlearn t={t}] loss_u={loss_u_avg:.4f} gnorm={gnorm:.2f} "
                    f"clip_scale={clip_scale:.3f} comp_norm={comp_norm_after:.2f}"
                )

        if (t + 1) % args.log_every == 0:
            print(
                f"[ticket t={t}] loss={total_loss:.4f} ema={ema:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if args.save_every > 0 and (t + 1) % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": t + 1,
                "args": asdict(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"PMP_ckpt_step{t+1}.pt"))

    return model


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    ap.add_argument("--mode", type=str, default="pmp", choices=["pmp", "base"])

    ap.add_argument("--out_dir", type=str, default="/root/autodl-tmp/result0.9")
    ap.add_argument("--train_steps", type=int, default=20000)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (ticket learning)")
    ap.add_argument("--unlearn_accum", type=int, default=8, help="Gradient accumulation steps (unlearning)")

    ap.add_argument("--rho", type=float, default=0.90)
    ap.add_argument("--tau", type=int, default=400)

    ap.add_argument("--eta", type=float, default=2e-5)
    ap.add_argument("--eta_u", type=float, default=2e-5)

    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.0)

    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--clip_ascent_norm", type=float, default=3.0)
    ap.add_argument("--clip_ticket_norm", type=float, default=1.0)

    ap.add_argument("--t_EB", type=int, default=500)
    ap.add_argument("--EB_iou_thresh", type=float, default=0.99)
    ap.add_argument("--EB_patience", type=int, default=5)

    ap.add_argument("--warmup_steps", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dataset_name", type=str, default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", type=str, default="default")
    ap.add_argument("--text_field", type=str, default="text")

    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--no_streaming", dest="streaming", action="store_false")
    ap.add_argument("--shuffle_buffer", type=int, default=50000)

    ap.add_argument("--log_unlearn_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=0)

    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ema_beta", type=float, default=0.98)

    return ap.parse_args()


def merge_args(cli, defaults: PMPArgs) -> PMPArgs:
    d = asdict(defaults)
    for k, v in vars(cli).items():
        if v is not None:
            d[k] = v
    return PMPArgs(**d)


def _force_save_all(model, tokenizer, args: PMPArgs, tag: str):
    os.makedirs(args.out_dir, exist_ok=True)

    weights_path = os.path.join(args.out_dir, f"{tag}_WEIGHTS_ONLY.pt")
    torch.save(model.state_dict(), weights_path)

    hf_dir = os.path.join(args.out_dir, f"{tag}_HF")
    os.makedirs(hf_dir, exist_ok=True)
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    model.config.to_json_file(os.path.join(args.out_dir, "config.json"))
    tokenizer.save_pretrained(args.out_dir)

    print(f"[SAVE] weights: {weights_path}")
    print(f"[SAVE] hf_dir : {hf_dir}")
    print(f"[SAVE] out_dir: {args.out_dir}")


def main():
    cli = parse_args()
    args = merge_args(cli, PMPArgs())

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = initialize_model(args, device)

    init_ckpt_path = os.path.join(args.out_dir, "_INIT_RANDOM_STATE.pt")
    torch.save(model.state_dict(), init_ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pre_loader, train_loader = make_stream_loaders(tokenizer, args)
    tag = "UNKNOWN"
    try:
        if args.mode == "base":
            tag = "BASE"
            print(">>> [MODE=BASE] Starting from random init...")
            model = base_train(model, train_loader, args, device)
        else:
            tag = "PMP"
            print(">>> [MODE=PMP] Stage I: Early-Bird discovery...")
            M = earlybird_discovery(model, pre_loader, args, device)

            init_sd = safe_torch_load_state_dict(init_ckpt_path, map_location="cpu")
            model.load_state_dict(init_sd, strict=True)

            save_mask_packed_atomic(M, os.path.join(args.out_dir, "PRIVATE_MASK_PACKED.pt"))

            print(">>> Stage II: PMP training...")
            model = pmp_train_stage2(model, train_loader, M, args, device)

    except KeyboardInterrupt:
        print("\n[WARN] KeyboardInterrupt received. Will force-save current model...")
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}. Will force-save current model...")
        raise
    finally:
        _force_save_all(model, tokenizer, args, tag)

    print("[DONE] Training finished & model saved.")


if __name__ == "__main__":
    main()
