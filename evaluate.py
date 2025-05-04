import torch
from models.model import InvisibleObjectDetector
from utils.metrics import calculate_map
from utils.nms import non_max_suppression
import numpy as np
from utils.config import Config

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_ground_truths = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            predictions = model(inputs)
            # 假设predictions是模型输出的检测框
            # 这里需要根据实际情况处理predictions和targets
            all_predictions.extend(predictions.cpu().numpy())
            all_ground_truths.extend(targets.cpu().numpy())
    predictions = np.array(all_predictions)
    ground_truths = np.array(all_ground_truths)
    map_score = calculate_map(predictions, ground_truths)
    print(f'mAP: {map_score:.4f}')

if __name__ == '__main__':
    config = Config('configs/model_config.yaml', 'configs/train_config.yaml')
    model = InvisibleObjectDetector(config)
    test_loader = ...  # 定义测试数据加载器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    evaluate_model(model, test_loader, device)
