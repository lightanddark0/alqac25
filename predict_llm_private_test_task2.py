import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_questions(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        q = item["text"]
        qid = item["question_id"]
        qtype = item.get("question_type", "")
        choices = item.get("choices", None)
        relevant_articles = item.get("relevant_articles", [])
        samples.append({
            "question_id": qid,
            "prompt": q,
            "question_type": qtype,
            "choices": choices,
            "relevant_articles": relevant_articles
        })
    return samples

def load_law_corpus(corpus_path):
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    # Tạo dict {(law_id, article_id): text}
    law_dict = {}
    for law in data:
        law_id = law.get("id")
        for article in law.get("articles", []):
            article_id = article.get("id")
            text = article.get("text", "")
            law_dict[(law_id, article_id)] = text
    return law_dict

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
        context = "[Context pháp lý liên quan]:\n" + "\n".join(context_parts) + "\n[Hết context]\n"
    else:
        context = ""
    return context

def build_prompt(q, law_dict):
    context = build_context(q, law_dict)
    qtype = q.get("question_type", "")
    text = q["prompt"]
    if qtype == "Đúng/Sai":
        prompt = (
            f"{context}Bạn là chuyên gia pháp luật. Hãy trả lời Đúng hoặc Sai cho câu sau, không cần giải thích thêm.\n"
            f"Câu hỏi: {text}\n"
            f"Đáp án:"
        )
    elif qtype == "Trắc nghiệm":
        choices = q.get("choices")
        if choices:
            choices_str = "\n".join([f"{k}: {v}" for k, v in choices.items()])
            prompt = (
                f"{context}Bạn là chuyên gia pháp luật. Hãy chọn đáp án đúng nhất (A, B, C hoặc D) cho câu hỏi sau, chỉ trả lời ký tự đáp án, không cần giải thích.\n"
                f"Câu hỏi: {text}\n"
                f"Các lựa chọn:\n{choices_str}\n"
                f"Đáp án:"
            )
        else:
            prompt = (
                f"{context}Bạn là chuyên gia pháp luật. Hãy chọn đáp án đúng nhất cho câu hỏi sau, chỉ trả lời ký tự đáp án, không cần giải thích.\n"
                f"Câu hỏi: {text}\n"
                f"Đáp án:"
            )
    elif qtype == "Tự luận":
        prompt = (
            f"{context}Bạn là chuyên gia pháp luật. Hãy trả lời ngắn gọn, chính xác và đầy đủ cho câu hỏi sau.\n"
            f"Câu hỏi: {text}\n"
            f"Đáp án:"
        )
    else:
        prompt = f"{context}Câu hỏi: {text}\nĐáp án:"
    return prompt

MODEL_NAME = "AITeamVN/Vi-Qwen2-1.5B-RAG"  # Đường dẫn model đã fine-tune (chỉnh lại nếu cần)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    questions = load_questions("alqac25_private_test_task2.json")
    law_dict = load_law_corpus("alqac25_law.json")

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None
    )
    model.eval()

    # Chuẩn bị prompt phù hợp từng loại câu hỏi, có context
    inputs = [build_prompt(q, law_dict) for q in questions]
    encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encodings["input_ids"],
            attention_mask=encodings["attention_mask"],
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Lưu kết quả
    results = [
        {"question_id": q["question_id"], "question": q["prompt"], "prediction": pred}
        for q, pred in zip(questions, predictions)
    ]
    with open("alqac25_private_test_task2_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Đã lưu kết quả dự đoán vào alqac25_private_test_task2_predictions.json") 