import numpy as np
from typing import List, Dict, Union

# @title
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.15)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=0.15)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=1))
    
def find_best(listdetect):

    result = sort_detections_by_criteria(listdetect)
    idx = result['sorted_indices'][0]
    best = result['sorted_detections'][0]

    #canvas = ellipse_annotator.annotate(listframes[idx], best)
    return best,idx
    

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """محاسبه IoU بین دو باکس"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def calculate_avg_overlap(detection) -> float:
    """محاسبه میانگین اورلپ برای یک detection"""
    boxes = detection.xyxy
    total_iou, count = 0, 0
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            total_iou += calculate_iou(boxes[i], boxes[j])
            count += 1
    return total_iou / count if count > 0 else 0

def sort_detections_by_criteria(detections: List) -> Dict[str, Union[List, int]]:
    """
    مرتب‌سازی detections بر اساس:
    1. تعداد bbox (نزولی)
    2. میانگین اورلپ (صعودی)
    
    خروجی:
    {
        'sorted_indices': لیست ایندکس‌ها از بهترین به بدترین (بر اساس لیست ورودی),
        'sorted_detections': لیست detections مرتب‌شده,
        'best_index': ایندکس بهترین detection در لیست اصلی
    }
    """
    if not detections:
        return {'sorted_indices': [], 'sorted_detections': [], 'best_index': None}
    
    # محاسبه معیارها برای هر detection
    det_metrics = []
    for idx, det in enumerate(detections):
        metrics = {
            'index': idx,  # ایندکس در لیست اصلی
            'num_boxes': len(det.xyxy),
            'avg_overlap': calculate_avg_overlap(det),
            'detection': det
        }
        det_metrics.append(metrics)
    
    # مرتب‌سازی با اولویت معیارها
    sorted_detections = sorted(
        det_metrics,
        key=lambda x: (-x['num_boxes'], x['avg_overlap'])
        #key=lambda x: (x['avg_overlap'])

    )
    
    # استخراج نتایج
    sorted_indices = [det['index'] for det in sorted_detections]
    best_index = sorted_indices[0] if sorted_indices else None
    
    return {
        'sorted_indices': sorted_indices,
        'sorted_detections': [det['detection'] for det in sorted_detections],
        'best_index': best_index
    }
    
