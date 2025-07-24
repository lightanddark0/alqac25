import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 1. Load luật corpus
def load_law_corpus(corpus_path):
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    law_dict = {}
    for law in data:
        law_id = law.get("id")
        for article in law.get("articles", []):
            article_id = article.get("id")
            text = article.get("text", "")
            law_dict[(law_id, article_id)] = text
    return law_dict

# 2. Build context cho từng câu hỏi
def build_context(q, law_dict):
    relevant = q.get("relevant_articles", [])
    context_parts = []
    for art in relevant:
        law_id = art.get("law_id")
        article_id = art.get("article_id")
        text = law_dict.get((law_id, article_id))
        if text:
            context_parts.append(f"- {law_id}, Điều {article_id}: {text}")
    if context_parts:
        context = "\n".join(context_parts)
    else:
        context = ""
    return context

# 3. Template prompt
template = '''Bạn là một chuyên gia về pháp luật
Chú ý các yêu cầu sau:
- Câu trả lời phải chính xác. 
- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
- Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.
- Không giải thích thêm.
- Không ghi lại ngữ cảnh và câu hỏi
Hãy trả lời câu hỏi dựa trên ngữ cảnh:
'''

def build_prompt(q, context):
    qtype = q.get("question_type", "")
    text = q["text"]
    if qtype == "Đúng/Sai":
        prompt = (
            f"Bạn là chuyên gia pháp luật. Chỉ trả lời Đúng hoặc Sai cho câu sau, không cần giải thích thêm, không nêu lý do.\n"
            f"Câu hỏi: {text}\n"
            f"Ngữ cảnh:{context}\n"
            f"Đáp án:"
        )
    elif qtype == "Trắc nghiệm":
        choices = q.get("choices")
        if choices:
            choices_str = "\n".join([f"{k}: {v}" for k, v in choices.items()])
            prompt = (
                f"Bạn là chuyên gia pháp luật. Chỉ trả lời đáp án đúng nhất (A, B, C hoặc D) cho câu hỏi sau, chỉ trả lời ký tự đáp án, không cần giải thích, không nêu lý do.\n"
                f"Câu hỏi: {text}\n"
                f"Các lựa chọn:\n{choices_str}\n"
                f"Ngữ cảnh:{context}\n"
                f"Đáp án:"
            )
    elif qtype == "Tự luận":
        prompt = (
            f"Bạn là chuyên gia pháp luật. Hãy trả lời ngắn gọn, chính xác, không nêu lý do.\n"
            f"Câu hỏi: {text}\n"
            f"Ngữ cảnh:{context}\n"
            f"Đáp án:"
        )

    return prompt

# 4. Load model & tokenizer
path = 'AITeamVN/GRPO-VI-Qwen2-1.5B-RAG'  # hoặc model HuggingFace bạn muốn
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
    device_map="auto",
    use_cache=True
)
tokenizer = AutoTokenizer.from_pretrained(path)

# 5. Đọc file câu hỏi và luật
with open("alqac25_test_split.json", encoding="utf-8") as f:
    questions = json.load(f)
law_dict = load_law_corpus("alqac25_law.json")
results = []
#questions = questions[:5]
for q in tqdm(questions):
    context = build_context(q, law_dict)
    #prompt = template.format(context=context, question=q["text"])
    prompt = [build_prompt(q, context)]
    # Nếu model là Qwen2/chat, có thể dùng chat template, nếu không thì chỉ cần encode prompt
    #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=8,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results.append({
        "question_id": q["question_id"],
        "question": q["text"],
        "prediction": response.strip().split("\n")[0]
    })

with open("alqac25_test_split_predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Đã lưu kết quả vào alqac25_test_split_predictions.json")

# --- Đoạn script tính accuracy ---
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
