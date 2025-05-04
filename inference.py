import torch
from models.model import InvisibleObjectDetector
from utils.nms import non_max_suppression
import numpy as np
import cv2

def preprocess_image(image_path, input_size=(416, 416)):
    # 预处理图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_size)
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # 添加批次维度
    image = torch.from_numpy(image).float() / 255.0  # 归一化
    return image

def inference(model, image_path, device):
    model.eval()
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        predictions = model(image)
    # 假设predictions是模型输出的检测框
    # 这里需要根据实际情况处理predictions
    boxes = predictions[0].cpu().numpy()
    scores = predictions[1].cpu().numpy()
    keep = non_max_suppression(boxes, scores)
    return boxes[keep], scores[keep]

if __name__ == '__main__':
    # 假设模型已经定义
    model = InvisibleObjectDetector(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image_path = 'path/to/your/image.jpg'
    boxes, scores = inference(model, image_path, device)
    print(f'Detected boxes: {boxes}')
    print(f'Scores: {scores}')
