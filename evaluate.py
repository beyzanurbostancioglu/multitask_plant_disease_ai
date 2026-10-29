"""Test kümesinde değerlendirme: doğruluklar, PDSA, Macro-F1, karışıklık matrisleri."""
import argparse, json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from class_map import PLANTS, DISEASE_KEYS, SEVERITY_TR, disease_name_tr
from dataset import LeafDataset
from model import MultiTaskNet


def cm_plot(y, p, names, path, title):
    cm = confusion_matrix(y, p, labels=range(len(names)), normalize="true")
    n = len(names); s = max(6, n * 0.35)
    fig, ax = plt.subplots(figsize=(s, s))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=6 if n > 12 else 9)
    ax.set_yticklabels(names, fontsize=6 if n > 12 else 9)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/exp1/best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = MultiTaskNet(ck["backbone"], pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    dl = DataLoader(LeafDataset(f"{a.data}/val.csv", ck["img"], False), 64,
                    num_workers=a.workers)
    P, Y = [[], [], []], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                outs = model(x.to(dev))
            for i, o in enumerate(outs):
                P[i].append(o.argmax(1).cpu())
            Y.append(y)
    P = [torch.cat(p).numpy() for p in P]; Y = torch.cat(Y).numpy()
    hs = Y[:, 2] >= 0
    ok = (P[0] == Y[:, 0]) & (P[1] == Y[:, 1]) & np.where(hs, P[2] == Y[:, 2], True)
    res = dict(
        n_test=int(len(Y)),
        acc_plant=float((P[0] == Y[:, 0]).mean()),
        acc_disease=float((P[1] == Y[:, 1]).mean()),
        acc_severity=float((P[2][hs] == Y[hs, 2]).mean()),
        pdsa=float(ok.mean()),
        macro_f1_plant=float(f1_score(Y[:, 0], P[0], average="macro")),
        macro_f1_disease=float(f1_score(Y[:, 1], P[1], average="macro")),
        macro_f1_severity=float(f1_score(Y[hs, 2], P[2][hs], average="macro")),
    )
    out = Path(a.ckpt).parent
    (out / "test_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))

    sev_names = [SEVERITY_TR[i] for i in range(3)]
    dis_names = [disease_name_tr(k) for k in DISEASE_KEYS]
    present = sorted(set(Y[:, 1]))
    (out / "report_disease.txt").write_text(classification_report(
        Y[:, 1], P[1], labels=present, target_names=[dis_names[i] for i in present],
        zero_division=0))
    cm_plot(Y[hs, 2], P[2][hs], sev_names, out / "cm_severity.png", "Şiddet")
    cm_plot(Y[:, 0], P[0], [p for p in PLANTS], out / "cm_plant.png", "Bitki türü")
    cm_plot(Y[:, 1], P[1], dis_names, out / "cm_disease.png", "Hastalık")
    print("Grafikler ve raporlar:", out)


if __name__ == "__main__":
    main()
