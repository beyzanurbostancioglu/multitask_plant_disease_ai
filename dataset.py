import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(size, train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.55, 1.0)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(0.25, 0.25, 0.2, 0.03),
            T.ToTensor(), T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class LeafDataset(Dataset):
    def __init__(self, csv_path, size=224, train=False):
        self.df = pd.read_csv(csv_path)
        self.tf = build_transforms(size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert("RGB")
        y = torch.tensor([r.plant, r.disease, r.severity], dtype=torch.long)
        return self.tf(img), y
