import numpy as np
import torch

def calculate_iou(box1, box2):
    # 计算两个框的IoU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    # 计算mAP
    ap = []
    for class_id in range(predictions.shape[1]):
        pred_boxes = predictions[predictions[:, class_id] > 0]
        gt_boxes = ground_truths[ground_truths[:, class_id] > 0]
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            ap.append(0)
            continue
        ious = np.array([[calculate_iou(pred, gt) for gt in gt_boxes] for pred in pred_boxes])
        max_ious = np.max(ious, axis=1)
        tp = np.sum(max_ious >= iou_threshold)
        fp = len(pred_boxes) - tp
        ap.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    return np.mean(ap)
