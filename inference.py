"""Eğitilmiş modelle tek görüntü analizi: bitki + hastalık + şiddet (model) + hastalıklı alan oranı (HSV)."""
import cv2
import numpy as np
import torch
from PIL import Image

from class_map import DISEASE_PLANT, DISEASE_KEYS, PLANTS, PLANTS_TR, SEVERITY_TR, disease_name_tr
from dataset import build_transforms
from model import MultiTaskNet
from severity import analyze, load_thresholds


class Predictor:
    def __init__(self, ckpt="runs/exp1/best.pt", device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        self.model = MultiTaskNet(ck["backbone"], pretrained=False).to(self.dev)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        self.tf = build_transforms(ck["img"], train=False)
        self.th = load_thresholds()
        # hastalık -> bitki uyumluluk maskesi (healthy her bitkiyle uyumlu)
        self.compat = torch.tensor([[dp is None or dp == p for dp in DISEASE_PLANT]
                                    for p in PLANTS], device=self.dev)

    @torch.no_grad()
    def predict(self, pil_img, with_ratio=True):
        x = self.tf(pil_img.convert("RGB")).unsqueeze(0).to(self.dev)
        lp, ld, ls = self.model(x)
        pp = lp.softmax(1)[0]; plant = int(pp.argmax())
        ld = ld.masked_fill(~self.compat[plant].unsqueeze(0), -1e4)   # tutarlılık: bitkiye ait olmayan hastalıkları kapat
        pd_ = ld.softmax(1)[0]; dis = int(pd_.argmax())
        ps = ls.softmax(1)[0]; sev = int(ps.argmax())
        if DISEASE_KEYS[dis] == "healthy":
            sev = 0
        res = dict(
            plant=PLANTS_TR[PLANTS[plant]], plant_conf=float(pp[plant]),
            disease=disease_name_tr(DISEASE_KEYS[dis]), disease_conf=float(pd_[dis]),
            severity=SEVERITY_TR[sev], severity_conf=float(ps[sev]),
            healthy=DISEASE_KEYS[dis] == "healthy",
            top_diseases={disease_name_tr(DISEASE_KEYS[i]): float(pd_[i]) for i in pd_.topk(3).indices},
        )
        if with_ratio:
            bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            r = analyze(bgr, self.th)
            res.update(ratio=r["ratio"], ratio_level=SEVERITY_TR[r["level"]],
                       overlay=cv2.cvtColor(r["overlay"], cv2.COLOR_BGR2RGB))
        return res
