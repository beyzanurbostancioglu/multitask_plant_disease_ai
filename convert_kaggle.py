import os
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
from class_map import CLASS_MAP

dataset_dir = r"C:\Users\beyza\Downloads\archive (2)\dataset"
out_dir = "data"
os.makedirs(out_dir, exist_ok=True)

def process_split(split_name):
    split_path = os.path.join(dataset_dir, split_name)
    records = []
    if not os.path.exists(split_path):
        return
    print(f"{split_name} seti işleniyor...")
    for class_id_str in tqdm(os.listdir(split_path)):
        class_folder = os.path.join(split_path, class_id_str)
        if not os.path.isdir(class_folder):
            continue
        try:
            cid = int(class_id_str)
        except ValueError:
            continue
        
        if cid not in CLASS_MAP:
            continue
        
        info = CLASS_MAP[cid]
        img_files = glob.glob(os.path.join(class_folder, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(class_folder, "*.[pP][nN][gG]"))
        
        for img_path in img_files:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                continue
            records.append({
             "path": img_path,
             "label_id": cid,
             "plant": info["plant_id"],
             "disease": info["disease_id"],
             "severity": info["severity_id"],
             "width": w,
             "height": h
})
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, f"{split_name}.csv"), index=False)
    print(f"{split_name}.csv oluşturuldu ({len(df)} örnek).")

for split in ["train", "val", "test"]:
    process_split(split)

# class_stats.csv oluşturma
stats = []
for cid, info in CLASS_MAP.items():
    stats.append({
        "class_id": cid,
        "plant_name": info["plant_name"],
        "disease_name": info["disease_name"],
        "severity_name": info["severity_name"]
    })
pd.DataFrame(stats).to_csv(os.path.join(out_dir, "class_stats.csv"), index=False)
print("class_stats.csv başarıyla oluşturuldu!")