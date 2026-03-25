"""
visual knowledge reward
"""


import os
import re
import json
import math
from datetime import datetime
import torch

from torchvision.ops.boxes import box_area


answer_pattern = re.compile(r'<answer>([\S\n\t\v ]*?)</answer>')
num = r'-?\d+(?:\.\d+)?'
bbox_patterns = [
    re.compile(rf'\(({num}),.*?({num})\),\(({num}),({num})\)'),
    re.compile(rf'\[({num}), ({num}), ({num}), ({num})\]'),
    re.compile(rf'\(({num}), ({num}), ({num}), ({num})\)'),
    re.compile(rf'\(({num}), ({num})\)\n?.*?\(({num}), ({num})\)'),
]
entity_pattern = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
)
bbox_threshold = 0.65

# ==============
# construct the knowledge paths
# ==============
visual_knowledge_path = "/path/to/your/knowledge_dir/results-visual_knowledge.json"
with open(visual_knowledge_path, "r") as f:
    visual_knowledge_dataset = json.load(f)
entity_to_visual_knowledge = {}
for data_entity_name in visual_knowledge_dataset:
    entity_to_visual_knowledge[data_entity_name] = visual_knowledge_dataset[data_entity_name]["knowledge_category"]




def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def get_bbox(ans):
    for i, pattern in enumerate(bbox_patterns):
        predict_bbox = re.findall(pattern, ans)
        if len(predict_bbox) != 0:
            try:
                predict_bbox = (float(predict_bbox[-1][0].replace('[', '').replace('x', '')), float(predict_bbox[-1][1]), float(predict_bbox[-1][2]), float(predict_bbox[-1][3]))
            except:
                predict_bbox = [0, 0, 0, 0]
            if sum(predict_bbox) < 4:
                predict_bbox = [c*1000 for c in predict_bbox]

            return predict_bbox, i+1
    
    return (0., 0., 0., 0.), 0


def iou_reward(data_source, solution_str, ground_truth, extra_info, **kwargs):
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    reward = 0.0
    
    gt_bbox, _ = get_bbox(ground_truth)

    gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
    answer = re.findall(answer_pattern, solution_str)
    if len(answer) > 0:
        pred_bbox, _ = get_bbox(answer[0])
        pred_bbox = torch.tensor(pred_bbox, dtype=torch.float32).view(-1, 4)
        iou, _ = box_iou(gt_bbox, pred_bbox)
        iou = iou.item()
        reward = iou if iou > bbox_threshold else 0.0
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        global_step = extra_info['global_steps']
        try:
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                f.write(f"Global_step: {global_step}\n")
                f.write(f"Ground_truth: {ground_truth}\n")
                f.write(f"Solution: {solution_str}\n")
        except:
            pass
    return reward


def visual_knowledge_iou_reward(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """
    knowledge category:
        - completely correct
        - mostly correct
        - partially correct
        - mostly incorrect
        - completely incorrect
    """
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    reward = 0.0
    iou_log = 0.0
    
    gt_bbox, _ = get_bbox(ground_truth)

    m = entity_pattern.search(ground_truth)
    if m:
        current_entity = m.group(1).strip()
    else:
        return 0.0

    knowledge = entity_to_visual_knowledge.get(current_entity, "completely incorrect")

    gt_bbox = torch.tensor(gt_bbox, dtype=torch.float32).view(-1, 4)
    answer = re.findall(answer_pattern, solution_str)
    if len(answer) > 0:
        pred_bbox, _ = get_bbox(answer[0])
        pred_bbox = torch.tensor(pred_bbox, dtype=torch.float32).view(-1, 4)
        iou, _ = box_iou(gt_bbox, pred_bbox)
        iou = iou.item()
        iou_log = iou
        if iou > bbox_threshold:
            if knowledge == "completely correct":
                reward = iou * 0.65
            elif knowledge == "mostly correct":
                reward = iou * 0.85
            elif knowledge == "partially correct":
                reward = iou
            elif knowledge == "mostly incorrect":
                reward = min(iou * 1.2, 1)
            else:
                reward = min(iou * 1.4, 1)
        else:
            if knowledge == "completely correct":
                reward = -0.6
            elif knowledge == "mostly correct":
                reward = -0.5
            elif knowledge == "partially correct":
                reward = -0.4
            elif knowledge == "mostly incorrect":
                reward = -0.2
            else:
                reward = 0.0
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        global_step = extra_info['global_steps']
        try:
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} IoU reward: {reward} -------------\n")
                f.write(f"Global_step: {global_step}\n")
                f.write(f"Ground_truth: {ground_truth}\n")
                f.write(f"Visual Knowledge: {knowledge}\n")
                f.write(f"Solution: {solution_str}\n")
        except:
            pass
    return reward




def format_reward(data_source, solution_str, ground_truth=None, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>[\S\n\t\v ]*?</think>\s*<answer>[\S\n\t\v ]*?</answer>"

    completion_content = solution_str
    match = re.match(pattern, completion_content)
    think_format_reward = 0.5 if match else 0.0
    
    bbox_format_reward = 0.0
    answer = re.findall(answer_pattern, completion_content)
    if len(answer) > 0:
        _, match_type = get_bbox(answer[0])
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            try:
                with open(log_path, "a") as f:
                    f.write(f"Match_type: {match_type}\n")
            except:
                pass
        if match_type == 1:
            bbox_format_reward = 0.5
        else:
            bbox_format_reward = 0.0
    else:
        bbox_format_reward = 0.0
            
    reward = bbox_format_reward + think_format_reward

    return reward


# the final reward function
def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    iou = visual_knowledge_iou_reward(data_source, solution_str, ground_truth, extra_info, **kwargs)
    fmt = format_reward(data_source, solution_str, **kwargs)

    total_reward = iou + fmt

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path, "a") as f:
            f.write(f"[Combined Reward] IoU={iou:.3f}, Format={fmt:.3f}, Total={total_reward:.3f}\n")

    return total_reward
