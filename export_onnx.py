"""ONNX'e aktarır ve PyTorch çıktısıyla karşılaştırır. python export_onnx.py --ckpt runs/exp1/best.pt"""
import argparse
import numpy as np, torch
from model import MultiTaskNet

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/exp1/best.pt"); ap.add_argument("--out", default="model.onnx")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu")
m = MultiTaskNet(ck["backbone"], pretrained=False); m.load_state_dict(ck["model"]); m.eval()
x = torch.randn(1, 3, ck["img"], ck["img"])
torch.onnx.export(m, x, a.out, input_names=["image"], output_names=["plant", "disease", "severity"],
                  dynamic_axes={"image": {0: "batch"}}, opset_version=17)
import onnxruntime as ort
o = ort.InferenceSession(a.out).run(None, {"image": x.numpy()})
with torch.no_grad():
    t = m(x)
print("Maks. fark:", [float(np.abs(o[i] - t[i].numpy()).max()) for i in range(3)])
print("Kaydedildi:", a.out)
