"""Komut satırı: python predict.py foto.jpg [--ckpt runs/exp1/best.pt]"""
import argparse
import cv2
from PIL import Image
from inference import Predictor

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("--ckpt", default="runs/exp1/best.pt")
a = ap.parse_args()
r = Predictor(a.ckpt).predict(Image.open(a.image))
print(f"Bitki    : {r['plant']} (%{r['plant_conf']*100:.1f})")
print(f"Hastalık : {r['disease']} (%{r['disease_conf']*100:.1f})")
print(f"Şiddet   : {r['severity']} (%{r['severity_conf']*100:.1f}) [model]")
print(f"Hastalıklı alan oranı: %{r['ratio']:.1f} -> {r['ratio_level']} [HSV]")
out = a.image.rsplit(".", 1)[0] + "_analiz.png"
cv2.imwrite(out, cv2.cvtColor(r["overlay"], cv2.COLOR_RGB2BGR)); print("Kaplama görseli:", out)
