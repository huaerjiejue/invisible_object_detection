import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DefaultTransform


class TemporalDetectionDataset(Dataset):
    def __init__(self, root_dir, num_frames=5, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform or DefaultTransform()
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for clip_name in sorted(os.listdir(self.root_dir)):
            clip_path = os.path.join(self.root_dir, clip_name)
            if not os.path.isdir(clip_path):
                continue
            label_path = os.path.join(clip_path, "labels.json")
            if not os.path.exists(label_path):
                continue

            frames = sorted([f for f in os.listdir(clip_path) if f.endswith(".png")])
            if len(frames) < self.num_frames:
                continue

            samples.append({
                "clip_dir": clip_path,
                "frames": frames[:self.num_frames],
                "label_path": label_path
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_dir = sample["clip_dir"]
        frames = sample["frames"]
        label_path = sample["label_path"]

        with open(label_path, "r") as f:
            label_data = json.load(f)

        imgs, boxes_all, classes_all = [], [], []

        for frame in frames:
            img_path = os.path.join(clip_dir, frame)
            image = Image.open(img_path).convert("L")
            boxes = []
            classes = []

            for obj in label_data.get(frame, []):
                boxes.append(torch.tensor(obj["bbox"], dtype=torch.float32))
                classes.append(int(obj["class"]))

            # 如果没有目标则填空
            boxes = torch.stack(boxes) if boxes else torch.zeros((0, 4))
            classes = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros((0,), dtype=torch.long)

            image, boxes = self.transform(image, boxes)

            imgs.append(image)
            boxes_all.append(boxes)
            classes_all.append(classes)

        imgs = torch.stack(imgs, dim=0)  # [T, 1, H, W]

        return imgs, boxes_all, classes_all
