"""AI Challenger 2018 - 61 sınıfın (plant, disease, severity) çözümlemesi."""

PLANTS = ["apple", "cherry", "corn", "grape", "citrus", "peach",
          "pepper", "potato", "strawberry", "tomato"]
PLANTS_TR = {"apple": "Elma", "cherry": "Kiraz", "corn": "Mısır", "grape": "Üzüm",
             "citrus": "Narenciye", "peach": "Şeftali", "pepper": "Biber",
             "potato": "Patates", "strawberry": "Çilek", "tomato": "Domates"}

DISEASES_TR = {
    "healthy": "Sağlıklı",
    "scab": "Uyuz (kabuk)", "frogeye": "Gri leke (frogeye)", "cedar_rust": "Sedir pası",
    "powdery_mildew": "Külleme", "gray_leaf_spot": "Gri yaprak lekesi", "rust": "Pas",
    "leaf_blight": "Yaprak yanıklığı", "mosaic_virus": "Mozaik virüsü",
    "black_rot": "Kara çürüklük", "ring_spot": "Halka leke", "brown_spot": "Kahverengi leke",
    "greening": "Sarı ejder (HLB)", "bacterial_spot": "Bakteriyel leke",
    "early_blight": "Erken yanıklık", "late_blight": "Geç yanıklık",
    "leaf_scorch": "Yaprak yanıklığı (scorch)", "leaf_mold": "Yaprak küfü",
    "target_spot": "Hedef leke", "septoria": "Septoria yaprak lekesi",
    "spider_mite": "Kırmızı örümcek zararı", "ylcv": "Sarı yaprak kıvırcıklık virüsü",
}
SEVERITY_TR = {0: "Sağlıklı", 1: "Genel/Orta", 2: "Ciddi/Ağır"}
SEVERITY_NAMES = ["healthy", "general", "serious"]


def _build():
    rows = []

    def add(plant, disease, sevs):
        for s in sevs:
            rows.append((plant, disease, s))

    add("apple", "healthy", [0]); add("apple", "scab", [1, 2])
    add("apple", "frogeye", [-1]); add("apple", "cedar_rust", [1, 2])
    add("cherry", "healthy", [0]); add("cherry", "powdery_mildew", [1, 2])
    add("corn", "healthy", [0]); add("corn", "gray_leaf_spot", [1, 2])
    add("corn", "rust", [1, 2]); add("corn", "leaf_blight", [1, 2])
    add("corn", "mosaic_virus", [-1])
    add("grape", "healthy", [0]); add("grape", "black_rot", [1, 2])
    add("grape", "ring_spot", [1, 2]); add("grape", "brown_spot", [1, 2])
    add("citrus", "healthy", [0]); add("citrus", "greening", [1, 2])
    add("peach", "healthy", [0]); add("peach", "scab", [1, 2])
    add("pepper", "healthy", [0]); add("pepper", "scab", [1, 2])
    add("potato", "healthy", [0]); add("potato", "early_blight", [1, 2])
    add("potato", "late_blight", [1, 2])
    add("strawberry", "healthy", [0]); add("strawberry", "leaf_scorch", [1, 2])
    add("tomato", "healthy", [0])
    for d in ["powdery_mildew", "scab", "early_blight", "late_blight",
              "leaf_mold", "target_spot", "septoria", "spider_mite", "ylcv"]:
        add("tomato", d, [1, 2])
    add("tomato", "mosaic_virus", [-1])
    return rows


CLASS_ROWS = _build()
assert len(CLASS_ROWS) == 61

DISEASE_KEYS = ["healthy"]
for p, d, s in CLASS_ROWS:
    k = d if d == "healthy" else f"{p}_{d}"
    if k not in DISEASE_KEYS:
        DISEASE_KEYS.append(k)

PLANT_TO_ID = {p: i for i, p in enumerate(PLANTS)}
DISEASE_TO_ID = {k: i for i, k in enumerate(DISEASE_KEYS)}
DISEASE_PLANT = [None if k == "healthy" else k.split("_", 1)[0] for k in DISEASE_KEYS]

# CLASS_MAP sözlüğü eklendi:
CLASS_MAP = {}
for cid, (p, d, s) in enumerate(CLASS_ROWS):
    key = d if d == "healthy" else f"{p}_{d}"
    CLASS_MAP[cid] = {
        "plant_name": p,
        "disease_name": d,
        "severity_name": SEVERITY_NAMES[s] if s in [0, 1, 2] else "none",
        "plant_id": PLANT_TO_ID[p],
        "disease_id": DISEASE_TO_ID[key],
        "severity_id": s
    }


def disease_name_tr(key):
    if key == "healthy":
        return "Sağlıklı"
    plant, dis = key.split("_", 1)
    return f"{PLANTS_TR[plant]} - {DISEASES_TR[dis]}"


def decode(class61):
    """61'li sınıf -> (plant_id, disease_id, severity)"""
    p, d, s = CLASS_ROWS[class61]
    key = d if d == "healthy" else f"{p}_{d}"
    return PLANT_TO_ID[p], DISEASE_TO_ID[key], s


if __name__ == "__main__":
    print(len(PLANTS), "bitki,", len(DISEASE_KEYS) - 1, "hastalık (+healthy)")