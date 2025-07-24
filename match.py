# --- Đoạn script tính accuracy ---
import json
with open("alqac25_test_split_predictions.json", encoding="utf-8") as f:
    preds = json.load(f)
with open("alqac25_test_split.json", encoding="utf-8") as f:
    questions = json.load(f)

# Tạo map từ question_id sang prediction
pred_map = {item["question_id"]: item["prediction"] for item in preds}

total = 0
correct = 0
for q in questions:
    qid = q["question_id"]
    answer = q.get("answer")
    if answer is not None and qid in pred_map:
        total += 1
        # So sánh đáp án, loại bỏ khoảng trắng dư thừa và chuẩn hóa chữ hoa/thường
        if str(pred_map[qid]).strip().lower() == str(answer).strip().lower():
            correct += 1

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
