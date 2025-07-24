import json
import numpy as np

# Đường dẫn file
PRED_FILE = 'alqac25_Task_1_top1.json'
GT_FILE = 'alqac25_private_test_task2.json'

# Đọc file
with open(PRED_FILE, encoding='utf-8') as f:
    pred_data = json.load(f)
with open(GT_FILE, encoding='utf-8') as f:
    gt_data = json.load(f)

precisions = []
recalls = []
f2s = []

for gt_item, pred_item in zip(gt_data, pred_data):
    gt_set = set(f"{a['law_id']}_{a['article_id']}" for a in gt_item['relevant_articles'])
    pred_set = set(f"{a['law_id']}_{a['article_id']}" for a in pred_item['relevant_articles'])
    inter = gt_set & pred_set
    precision = len(inter) / len(pred_set) if pred_set else 0.0
    recall = len(inter) / len(gt_set) if gt_set else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0
    precisions.append(precision)
    recalls.append(recall)
    f2s.append(f2)

print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall:    {np.mean(recalls):.4f}")
print(f"F2:        {np.mean(f2s):.4f}") 