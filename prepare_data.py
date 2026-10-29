"""AI Challenger 2018 verisini işler: etiketleri çözer, küçültür, train/val/test böler.

Kullanım:
  python prepare_data.py --root D:/veri/ai_challenger --out data --size 320

--root altında (iç içe de olsa) şunlar aranır:
  * *annotations*.json  (ör. AgriculturalDisease_train_annotations.json)
  * görüntüler (.jpg) - dosya adı JSON'daki image_id ile eşleşir
"""
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from class_map import CLASS_ROWS, decode


def index_images(root):
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                idx[f] = os.path.join(dp, f)
    return idx


def load_annotations(root):
    recs = []
    for p in Path(root).rglob("*.json"):
        if "annotation" not in p.name.lower():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list) and data and "disease_class" in data[0]:
            print(f"  {p.name}: {len(data)} kayıt")
            recs += [(d["image_id"], int(d["disease_class"])) for d in data]
    return recs


def _resize(args):
    src, dst, size = args
    try:
        if os.path.exists(dst):
            return True
        im = Image.open(src).convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(dst, quality=92)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Ham veri klasörü")
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=320, help="Uzun kenar (px); 0 = küçültme")
    ap.add_argument("--min_samples", type=int, default=20,
                    help="Bundan az örnekli sınıflar atılır (ör. 3 örnekli 'Canker')")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    print("Etiketler okunuyor...")
    recs = load_annotations(a.root)
    if not recs:
        raise SystemExit("Annotation JSON bulunamadı. --root yolunu kontrol et.")
    print("Görüntüler indeksleniyor...")
    img_idx = index_images(a.root)

    rows, missing = [], 0
    for image_id, c in recs:
        name = os.path.basename(image_id)
        if name not in img_idx or not (0 <= c < 61):
            missing += 1
            continue
        pid, did, sev = decode(c)
        rows.append(dict(image=name, src=img_idx[name], class61=c,
                         plant=pid, disease=did, severity=sev))
    df = pd.DataFrame(rows).drop_duplicates("image")
    print(f"Eşleşen: {len(df)} | eksik/geçersiz: {missing}")

    # sınıf istatistiği
    cnt = df.class61.value_counts().reindex(range(61), fill_value=0)
    stats = pd.DataFrame({"class61": range(61), "count": cnt.values,
                          "plant": [r[0] for r in CLASS_ROWS],
                          "disease": [r[1] for r in CLASS_ROWS],
                          "severity": [r[2] for r in CLASS_ROWS]})
    stats.to_csv(out / "class_stats.csv", index=False)
    print("Boş sınıflar:", list(stats[stats["count"] == 0].class61))
    rare = cnt[(cnt > 0) & (cnt < a.min_samples)]
    if len(rare):
        print(f"Az örnekli sınıflar atılıyor (<{a.min_samples}):", dict(rare))
    df = df[df.class61.map(cnt) >= a.min_samples].copy()

    # yeniden boyutlandırma (paralel)
    print("Görüntüler küçültülüyor/kopyalanıyor...")
    dsts = [str(out / "images" / n) for n in df.image]
    if a.size > 0:
        jobs = [(s, d, a.size) for s, d in zip(df.src, dsts)]
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as ex:
            ok = list(tqdm(ex.map(_resize, jobs, chunksize=64), total=len(jobs)))
        df = df[ok]
        dsts = [str(out / "images" / n) for n in df.image]
    else:
        dsts = list(df.src)
    df["path"] = dsts
    df = df.drop(columns="src")

    # stratified 80/10/10
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df.class61, random_state=a.seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.class61, random_state=a.seed)
    for n, d in [("train", tr), ("val", va), ("test", te)]:
        d.to_csv(out / f"{n}.csv", index=False)
        print(f"{n}: {len(d)}")
    print("Bitti ->", out.resolve())


if __name__ == "__main__":
    main()
