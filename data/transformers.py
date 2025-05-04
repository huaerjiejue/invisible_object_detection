import torchvision.transforms as T
import torch


class DefaultTransform:
    def __init__(self, image_size=(256, 256)):
        self.resize = T.Resize(image_size)
        self.to_tensor = T.ToTensor()

    def __call__(self, image, boxes):
        orig_w, orig_h = image.size
        image = self.resize(image)
        image = self.to_tensor(image)  # [1, H, W]

        # 缩放 boxes
        new_w, new_h = image.shape[2], image.shape[1]
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        if boxes.numel() > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return image, boxes
