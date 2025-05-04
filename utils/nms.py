import numpy as np

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # 非极大值抑制
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        if indices.size == 1:
            break
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    return keep
