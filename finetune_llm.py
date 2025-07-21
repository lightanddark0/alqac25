import unsloth
import json
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch


# 1. Import Unsloth
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import LoraConfig

MODEL_NAME = "AITeamVN/GRPO-VI-Qwen2-7B-RAG"  # Ví dụ: model đã hỗ trợ Unsloth, bạn thay bằng model phù hợp
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_qa_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        q = item["text"]
        a = item.get("answer", "")
        samples.append({"prompt": q, "response": a})
    return samples

def preprocess(samples, tokenizer, max_length=512):
    inputs = [f"Câu hỏi: {s['prompt']}\nĐáp án:" for s in samples]
    targets = [s['response'] for s in samples]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    # 1. Load data
    train_samples = load_qa_data("alqac25_train_split.json")
    test_samples = load_qa_data("alqac25_test_split.json")
    try:
        private_test_samples = load_qa_data("alqac25_private_test_task2.json")
    except Exception:
        private_test_samples = []

    # 2. LoRA config
    lora_config = LoraConfig(
        r=16,  # Số rank, càng nhỏ càng nhẹ, 8-16 là hợp lý
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",  # LoRA cho language model
        target_modules=["q_proj", "v_proj"]  # Tùy model, có thể là ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # 3. Load model & tokenizer with Unsloth + LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
        device_map="auto",
        lora_config=lora_config
    )

    # 4. Chuẩn hóa dữ liệu
    train_dataset = Dataset.from_dict(preprocess(train_samples, tokenizer))
    test_dataset = Dataset.from_dict(preprocess(test_samples, tokenizer))

    # 5. Training arguments
    args = TrainingArguments(
        output_dir="viQwen-lora",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-5,
        fp16=True if DEVICE == "cuda" else False,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7. Fine-tune
    trainer.train()

    # 8. Lưu lại adapter LoRA (chỉ lưu phần LoRA, rất nhẹ)
    model.save_pretrained("viQwen-lora-adapter")
    tokenizer.save_pretrained("viQwen-lora-adapter")

    # 9. Đánh giá
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # 10. Dự đoán trên file private test (nếu có)
    if private_test_samples:
        inputs = [f"Câu hỏi: {s['prompt']}\nĐáp án:" for s in private_test_samples]
        encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                max_new_tokens=64,
                do_sample=False
            )
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        with open("viQwen_private_test_predictions.json", "w", encoding="utf-8") as f:
            json.dump([
                {"question": s["prompt"], "prediction": pred}
                for s, pred in zip(private_test_samples, predictions)
            ], f, ensure_ascii=False, indent=2)
        print("Đã lưu kết quả dự đoán vào viQwen_private_test_predictions.json")

if __name__ == "__main__":
    print(f'Using device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    main()