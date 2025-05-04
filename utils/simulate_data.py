from PIL import Image, ImageDraw
import os, json, random


def generate_sequence(out_dir, clip_id, num_frames=5, image_size=(256, 256), num_objects=3):
    os.makedirs(out_dir, exist_ok=True)
    clip_path = os.path.join(out_dir, f"clip_{clip_id:04d}")
    os.makedirs(clip_path, exist_ok=True)

    labels = {}
    W, H = image_size

    # 初始化目标位置和速度
    objects = []
    for _ in range(num_objects):
        x, y = random.randint(20, W-60), random.randint(20, H-60)
        w, h = random.randint(20, 40), random.randint(20, 40)
        vx, vy = random.randint(-3, 3), random.randint(-3, 3)
        cls = random.randint(0, 4)
        objects.append({"x": x, "y": y, "w": w, "h": h, "vx": vx, "vy": vy, "class": cls})

    # 每帧更新位置，保存图像和bbox
    for t in range(num_frames):
        img = Image.new("L", (W, H), color=100)
        draw = ImageDraw.Draw(img)
        label_list = []

        for obj in objects:
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

            x1 = max(0, obj["x"])
            y1 = max(0, obj["y"])
            x2 = min(W, obj["x"] + obj["w"])
            y2 = min(H, obj["y"] + obj["h"])

            draw.rectangle([x1, y1, x2, y2], fill=200)
            label_list.append({"bbox": [x1, y1, x2, y2], "class": obj["class"]})

        fname = f"{t:04d}.png"
        img.save(os.path.join(clip_path, fname))
        labels[fname] = label_list

    with open(os.path.join(clip_path, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)


def generate_dataset(out_dir, num_clips=100):
    for i in range(num_clips):
        generate_sequence(out_dir, i)


if __name__ == "__main__":
    generate_dataset("data/simulated", num_clips=200)
