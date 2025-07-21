import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
# Thêm import cho LoRA/PEFT
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load corpus luật

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

# 2. Load dữ liệu huấn luyện

def load_qa_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        q = item["text"]
        a = item.get("answer", "")
        relevant_articles = item.get("relevant_articles", [])
        qtype = item.get("question_type", "")
        choices = item.get("choices", None)
        samples.append({
            "prompt": q,
            "response": a,
            "relevant_articles": relevant_articles,
            "question_type": qtype,
            "choices": choices
        })
    return samples

# 3. Build context cho từng câu hỏi

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

# 4. Build prompt cho từng loại câu hỏi

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

# 5. Chuẩn bị dữ liệu cho finetune

def prepare_finetune_samples(samples, law_dict):
    prompts = [build_prompt(q, law_dict) for q in samples]
    responses = [q["response"] for q in samples]
    return {"prompt": prompts, "response": responses}

if __name__ == "__main__":
    # Đường dẫn model base và dữ liệu
    MODEL_NAME = "AITeamVN/Vi-Qwen2-1.5B-RAG"  # hoặc model HuggingFace khác
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    print("Đang load dữ liệu...")
    law_dict = load_law_corpus("alqac25_law.json")
    train_samples = load_qa_data("alqac25_train_split.json")

    # Tối ưu tốc độ: chỉ lấy 1000 mẫu đầu để thử nghiệm nhanh (bạn có thể tăng lên nếu đủ RAM)
    train_samples = train_samples[:1000]

    print("Đang chuẩn bị dữ liệu huấn luyện...")
    data = prepare_finetune_samples(train_samples, law_dict)
    train_dataset = Dataset.from_dict(data)

    print("Đang load model và tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None
    )

    # Thêm LoRA/PEFT
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Tùy model, có thể thêm "k_proj", "o_proj"
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # 6. Tiền xử lý dữ liệu
    def preprocess(samples):
        inputs = samples["prompt"]
        targets = samples["response"]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Đang tokenize dữ liệu...")
    tokenized_dataset = train_dataset.map(preprocess, batched=True)

    # 7. Training arguments (tối ưu tốc độ)
    args = TrainingArguments(
        output_dir="finetuned-llm-with-context-lora",
        per_device_train_batch_size=1,  # nhỏ để tiết kiệm RAM
        num_train_epochs=1,
        learning_rate=2e-4,  # tăng learning rate để convergence nhanh hơn
        fp16=True if DEVICE == "cuda" else False,
        save_strategy="no",  # không lưu checkpoint giữa chừng
        logging_steps=10,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Bắt đầu huấn luyện với LoRA...")
    trainer.train()

    print("Lưu adapter LoRA đã fine-tune...")
    model.save_pretrained("finetuned-llm-with-context-lora")
    tokenizer.save_pretrained("finetuned-llm-with-context-lora")
    print("Đã lưu adapter vào finetuned-llm-with-context-lora/")
    print("Nếu chưa cài peft: pip install peft") 