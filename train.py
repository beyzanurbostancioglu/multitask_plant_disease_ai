"""Çok görevli eğitim. RTX 4050 (6 GB) için varsayılanlar: efficientnet_b0, 224px, batch 32, AMP."""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from class_map import PLANTS, DISEASE_KEYS
from dataset import LeafDataset
from losses import FocalLoss, class_weights
from model import MultiTaskNet


@torch.no_grad()
def run_eval(model, loader, dev):
    model.eval()
    P, Y = [[], [], []], []
    for x, y in loader:
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            outs = model(x)
        for i, o in enumerate(outs):
            P[i].append(o.argmax(1).cpu())
        Y.append(y)
    P = [torch.cat(p).numpy() for p in P]
    Y = torch.cat(Y).numpy()
    ok_p, ok_d = P[0] == Y[:, 0], P[1] == Y[:, 1]
    has_s = Y[:, 2] >= 0
    ok_s = np.where(has_s, P[2] == Y[:, 2], True)
    return dict(
        acc_plant=ok_p.mean(), acc_disease=ok_d.mean(),
        acc_sev=(P[2][has_s] == Y[has_s, 2]).mean(),
        pdsa=(ok_p & ok_d & ok_s).mean(),
        f1_disease=f1_score(Y[:, 1], P[1], average="macro"),
        f1_sev=f1_score(Y[has_s, 2], P[2][has_s], average="macro"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--w_plant", type=float, default=0.5)
    ap.add_argument("--w_disease", type=float, default=1.0)
    ap.add_argument("--w_sev", type=float, default=1.0)
    a = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cihaz:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "(GPU YOK - çok yavaş olur!)")
    torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tr_ds = LeafDataset(f"{a.data}/train.csv", a.img, True)
    va_ds = LeafDataset(f"{a.data}/val.csv", a.img, False)
    kw = dict(num_workers=a.workers, pin_memory=True,
              persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    tr_dl = DataLoader(tr_ds, a.bs, shuffle=True, drop_last=True, **kw)
    va_dl = DataLoader(va_ds, a.bs * 2, shuffle=False, **kw)

    model = MultiTaskNet(a.backbone).to(dev).to(memory_format=torch.channels_last)
    df = tr_ds.df
    lab = lambda c: torch.tensor(df[c].values)
    crit_p = FocalLoss(1.0, class_weights(lab("plant"), len(PLANTS)).to(dev))
    crit_d = FocalLoss(2.0, class_weights(lab("disease"), len(DISEASE_KEYS)).to(dev))
    crit_s = FocalLoss(2.0, class_weights(lab("severity"), 3).to(dev))

    heads = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    body = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr},
                             {"params": heads, "lr": a.lr * 3}], weight_decay=a.wd)
    steps_ep = len(tr_dl); total = a.epochs * steps_ep; warm = steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best, bad, log = -1, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        pbar = tqdm(tr_dl, desc=f"Epoch {ep}/{a.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                op, od, os_ = model(x)
                loss = (a.w_plant * crit_p(op.float(), y[:, 0]) +
                        a.w_disease * crit_d(od.float(), y[:, 1]) +
                        a.w_sev * crit_s(os_.float(), y[:, 2]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); pbar.set_postfix(loss=f"{loss.item():.3f}")
        m = run_eval(model, va_dl, dev)
        m.update(epoch=ep, train_loss=run / steps_ep, sec=time.time() - t0)
        log.append(m); pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        print(f"ep{ep:02d} loss={m['train_loss']:.3f} | bitki={m['acc_plant']:.4f} "
              f"hastalık={m['acc_disease']:.4f} şiddet={m['acc_sev']:.4f} "
              f"PDSA={m['pdsa']:.4f} F1(h)={m['f1_disease']:.4f} | {m['sec']:.0f}s")
        score = m["pdsa"] + 0.5 * m["f1_disease"]
        if score > best:
            best, bad = score, 0
            torch.save(dict(model=model.state_dict(), backbone=a.backbone, img=a.img,
                            plants=PLANTS, diseases=DISEASE_KEYS, metrics=m), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping."); break
    print("En iyi model:", out / "best.pt")


if __name__ == "__main__":
    main()   # Windows'ta DataLoader worker'ları için bu koruma şart
