"""Demo arayüzü: python app.py  -> tarayıcıda http://127.0.0.1:7860"""
import argparse
import gradio as gr
from inference import Predictor

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="runs/exp1/best.pt")
pred = Predictor(ap.parse_args().ckpt)


def run(img):
    if img is None:
        return "Görüntü yükleyin.", None, None
    r = pred.predict(img)
    txt = (f"**Bitki:** {r['plant']} (%{r['plant_conf']*100:.0f})\n\n"
           f"**Durum:** {'Sağlıklı' if r['healthy'] else r['disease']} (%{r['disease_conf']*100:.0f})\n\n"
           f"**Şiddet (model):** {r['severity']} (%{r['severity_conf']*100:.0f})\n\n"
           f"**Hastalıklı alan oranı (görüntü işleme):** %{r['ratio']:.1f} → {r['ratio_level']}")
    return txt, r["top_diseases"], r["overlay"]


gr.Interface(run, gr.Image(type="pil", label="Yaprak fotoğrafı"),
             [gr.Markdown(label="Sonuç"), gr.Label(label="En olası 3 hastalık"),
              gr.Image(label="Hastalıklı bölge (kırmızı)")],
             title="Bitki Hastalığı ve Şiddet Tespiti", flagging_mode="never").launch()
