# unsolth_finetune.py
# Fine-tuning với Unsloth cho ALQAC 2025

import json
import os
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import wandb
except ImportError as e:
    print(f"Thiếu thư viện: {e}")
    print("Chạy: pip install unsloth trl wandb")
    exit(1)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ALQACQuestion:
    """Cấu trúc dữ liệu cho câu hỏi ALQAC"""
    question_id: str
    question_type: str
    text: str
    relevant_articles: List[Dict[str, str]]
    answer: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ALQACQuestion':
        return cls(
            question_id=data['question_id'],
            question_type=data['question_type'],
            text=data['text'],
            relevant_articles=data['relevant_articles'],
            answer=data['answer']
        )

class ALQACDataProcessor:
    """Xử lý dữ liệu ALQAC cho fine-tuning"""
    
    def __init__(self, corpus_file: str = "alqac25_law.json"):
        self.corpus_file = corpus_file
        self.legal_corpus = self._load_corpus()
    
    def _load_corpus(self) -> Dict[str, str]:
        """Tải corpus pháp luật"""
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                corpus_data = json.load(f)
            
            corpus_dict = {}
            for law in corpus_data:
                law_id = law.get('id', '')
                articles = law.get('articles', [])
                for article in articles:
                    article_id = article.get('id', '')
                    content = article.get('text', '')
                    doc_id = f"{law_id}_{article_id}"
                    corpus_dict[doc_id] = content
            
            logger.info(f"Đã tải {len(corpus_dict)} văn bản pháp luật")
            return corpus_dict
            
        except Exception as e:
            logger.error(f"Lỗi khi tải corpus: {e}")
            return {}
    
    def build_context(self, relevant_articles: List[Dict[str, str]]) -> str:
        """Xây dựng context từ các văn bản liên quan"""
        if not relevant_articles:
            return "Không có tài liệu liên quan."
        
        context_parts = []
        for i, article in enumerate(relevant_articles, 1):
            law_id = article.get('law_id', '')
            article_id = article.get('article_id', '')
            doc_id = f"{law_id}_{article_id}"
            
            if doc_id in self.legal_corpus:
                content = self.legal_corpus[doc_id]
                context_parts.append(
                    f"Tài liệu {i} ({law_id} - Điều {article_id}):\n{content}\n"
                )
        
        return "\n".join(context_parts) if context_parts else "Không có tài liệu liên quan."
    
    def format_for_training(self, questions: List[ALQACQuestion]) -> List[Dict[str, str]]:
        """Format dữ liệu cho fine-tuning"""
        training_data = []
        
        for question in questions:
            # Xây dựng context
            context = self.build_context(question.relevant_articles)
            
            # Tạo prompt dựa trên loại câu hỏi
            if question.question_type == "Đúng/Sai":
                prompt = f"""Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi sau đây dựa trên các tài liệu pháp luật được cung cấp.

Câu hỏi (Đúng/Sai): {question.text}

Tài liệu tham khảo:
{context}

Yêu cầu:
- Trả lời chỉ "Đúng" hoặc "Sai"
- Giải thích ngắn gọn lý do

Trả lời:"""
                
            elif question.question_type == "Trắc nghiệm":
                choices = question.choices if hasattr(question, 'choices') else {}
                choices_text = ""
                for key, value in choices.items():
                    choices_text += f"{key}. {value}\n"
                
                prompt = f"""Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi sau đây dựa trên các tài liệu pháp luật được cung cấp.

Câu hỏi (Trắc nghiệm): {question.text}

Các lựa chọn:
{choices_text}

Tài liệu tham khảo:
{context}

Yêu cầu:
- Trả lời chỉ một trong các lựa chọn A, B, C, D
- Giải thích ngắn gọn lý do chọn

Trả lời:"""
                
            else:  # Tự luận
                prompt = f"""Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi sau đây dựa trên các tài liệu pháp luật được cung cấp.

Câu hỏi (Tự luận): {question.text}

Tài liệu tham khảo:
{context}

Yêu cầu:
- Trả lời chi tiết và đầy đủ
- Trích dẫn các điều luật liên quan
- Giải thích rõ ràng logic pháp lý

Trả lời:"""
            
            # Tạo completion
            completion = question.answer
            
            training_data.append({
                "prompt": prompt,
                "completion": completion
            })
        
        return training_data

class UnslothFineTuner:
    """Fine-tuning với Unsloth"""
    
    def __init__(self, 
                 model_name: str = "unsloth/llama-3-8b-bnb-4bit",
                 max_seq_length: int = 2048,
                 dtype: torch.dtype = torch.bfloat16,
                 load_in_4bit: bool = True):
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model(self):
        """Tải model với Unsloth"""
        logger.info(f"Đang tải model {self.model_name} với Unsloth...")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
            )
            
            # Thêm PEFT adapter
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            
            logger.info("Đã tải model thành công")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {e}")
            raise
    
    def prepare_dataset(self, training_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Chuẩn bị dataset cho training"""
        formatted_data = []
        
        for item in training_data:
            # Format theo chuẩn Unsloth
            formatted_item = {
                "text": f"{item['prompt']}{item['completion']}"
            }
            formatted_data.append(formatted_item)
        
        return formatted_data
    
    def train(self, 
              training_data: List[Dict[str, str]],
              output_dir: str = "./alqac_finetuned_model",
              num_epochs: int = 3,
              batch_size: int = 2,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100,
              logging_steps: int = 10,
              save_steps: int = 500,
              eval_steps: int = 500,
              gradient_accumulation_steps: int = 4):
        """Fine-tune model"""
        
        if not self.model:
            self.load_model()
        
        # Chuẩn bị dataset
        formatted_data = self.prepare_dataset(training_data)
        
        # Tạo training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
        )
        
        # Tạo trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=formatted_data,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        
        # Bắt đầu training
        logger.info("Bắt đầu fine-tuning...")
        self.trainer.train()
        
        # Lưu model
        logger.info(f"Lưu model vào {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Fine-tuning hoàn thành!")
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Sinh câu trả lời từ model đã fine-tune"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model chưa được tải")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Loại bỏ prompt khỏi response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Lỗi khi sinh câu trả lời: {e}")
            return "Không thể sinh câu trả lời"

def main():
    """Hàm main để chạy fine-tuning"""
    
    # Cấu hình
    config = {
        'model_name': 'unsloth/llama-3-8b-bnb-4bit',
        'max_seq_length': 2048,
        'num_epochs': 3,
        'batch_size': 2,
        'learning_rate': 2e-4,
        'output_dir': './alqac_finetuned_model'
    }
    
    try:
        # Tải dữ liệu training
        logger.info("Đang tải dữ liệu training...")
        with open('alqac25_train_split.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        questions = [ALQACQuestion.from_dict(q) for q in training_data]
        logger.info(f"Đã tải {len(questions)} câu hỏi training")
        
        # Xử lý dữ liệu
        processor = ALQACDataProcessor()
        formatted_data = processor.format_for_training(questions)
        logger.info(f"Đã format {len(formatted_data)} mẫu training")
        
        # Fine-tuning
        finetuner = UnslothFineTuner(
            model_name=config['model_name'],
            max_seq_length=config['max_seq_length']
        )
        
        finetuner.train(
            training_data=formatted_data,
            output_dir=config['output_dir'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
        
        # Test model
        logger.info("Testing model...")
        test_prompt = """Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi sau đây dựa trên các tài liệu pháp luật được cung cấp.

Câu hỏi (Đúng/Sai): Hợp đồng không bắt buộc phải có phụ lục kèm theo, đúng hay sai?

Tài liệu tham khảo:
Tài liệu 1 (Bộ luật dân sự - Điều 403): Hợp đồng có thể có phụ lục kèm theo để quy định chi tiết một số điều khoản của hợp đồng.

Yêu cầu:
- Trả lời chỉ "Đúng" hoặc "Sai"
- Giải thích ngắn gọn lý do

Trả lời:"""
        
        answer = finetuner.generate_answer(test_prompt)
        print(f"\nTest Answer: {answer}")
        
        print("\nFine-tuning hoàn thành thành công!")
        print(f"Model đã được lưu vào {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy fine-tuning: {e}")
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main() 