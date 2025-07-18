# bge_m3_alqac.py - Sử dụng BGE-M3 cho ALQAC Retrieval

import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import torch
from sentence_transformers import SentenceTransformer
import faiss

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ALQACQuestion:
    """Cấu trúc dữ liệu cho câu hỏi ALQAC"""
    question_id: str
    question_type: str  # "Đúng/Sai", "Trắc nghiệm", "Tự luận"
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

@dataclass
class LegalDocument:
    """Cấu trúc dữ liệu cho văn bản pháp luật"""
    law_id: str
    article_id: str
    content: str
    
    def get_id(self) -> str:
        return f"{self.law_id}_{self.article_id}"

@dataclass
class RetrievalResult:
    """Kết quả truy xuất"""
    document_id: str
    score: float
    document: LegalDocument

class BGE_M3Retriever:
    """Retriever sử dụng BGE-M3 model"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.documents = []
        self.embeddings = None
        self.index = None
        
        self._load_model()
    
    def _load_model(self):
        """Tải BGE-M3 model"""
        try:
            logger.info(f"Đang tải BGE-M3 model từ {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Đã tải BGE-M3 thành công")
        except Exception as e:
            logger.error(f"Lỗi khi tải BGE-M3: {e}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode danh sách văn bản thành embeddings
        
        Args:
            texts: Danh sách văn bản
            batch_size: Kích thước batch
            
        Returns:
            Ma trận embeddings
        """
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Lỗi khi encode texts: {e}")
            raise
    
    def build_index(self, documents: List[LegalDocument]):
        """
        Xây dựng FAISS index cho documents
        
        Args:
            documents: Danh sách documents
        """
        logger.info("Đang xây dựng FAISS index với BGE-M3...")
        
        self.documents = documents
        
        # Tạo texts cho encoding
        texts = [doc.content for doc in documents]
        
        # Encode documents
        self.embeddings = self.encode_texts(texts)
        
        # Tạo FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product cho cosine similarity
        
        # Thêm embeddings vào index
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Đã xây dựng index cho {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Truy xuất documents sử dụng BGE-M3
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            
        Returns:
            Danh sách kết quả được sắp xếp theo similarity
        """
        if self.index is None:
            raise ValueError("Chưa xây dựng index")
        
        # Encode query
        query_embedding = self.encode_texts([query])
        
        # Tìm kiếm trong index
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Tạo kết quả
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                result = RetrievalResult(
                    document_id=self.documents[idx].get_id(),
                    score=float(score),
                    document=self.documents[idx]
                )
                results.append(result)
        
        return results
    
    def save_index(self, filepath: str):
        """Lưu index ra file"""
        if self.index is not None:
            faiss.write_index(self.index, filepath)
            logger.info(f"Đã lưu index vào {filepath}")
    
    def load_index(self, filepath: str, documents: List[LegalDocument]):
        """Tải index từ file"""
        if Path(filepath).exists():
            self.index = faiss.read_index(filepath)
            self.documents = documents
            logger.info(f"Đã tải index từ {filepath}")
        else:
            logger.warning(f"File index không tồn tại: {filepath}")

class ALQACDataLoader:
    """Lớp để tải và xử lý dữ liệu ALQAC"""
    
    def __init__(self):
        self.questions: List[ALQACQuestion] = []
        self.legal_corpus: List[LegalDocument] = []
        
    def load_questions(self, json_file_path: str) -> List[ALQACQuestion]:
        """Tải câu hỏi từ file JSON"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                questions_data = [data]
            elif isinstance(data, list):
                questions_data = data
            else:
                raise ValueError("Định dạng file JSON không hợp lệ")
            
            self.questions = [ALQACQuestion.from_dict(q) for q in questions_data]
            logger.info(f"Đã tải {len(self.questions)} câu hỏi từ {json_file_path}")
            
            return self.questions
            
        except Exception as e:
            logger.error(f"Lỗi khi tải file JSON: {e}")
            raise
    
    def load_legal_corpus(self, corpus_path: str) -> List[LegalDocument]:
        """Tải corpus văn bản pháp luật"""
        try:
            if not Path(corpus_path).exists():
                logger.warning(f"File corpus không tồn tại: {corpus_path}")
                logger.info("Tạo corpus mẫu...")
                self.legal_corpus = self._create_sample_corpus()
            else:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    corpus_data = json.load(f)
                
                self.legal_corpus = []
                for law in corpus_data:
                    law_id = law.get('id', '')
                    articles = law.get('articles', [])
                    for article in articles:
                        article_id = article.get('id', '')
                        content = article.get('text', '')
                        self.legal_corpus.append(
                            LegalDocument(
                                law_id=law_id,
                                article_id=article_id,
                                content=content
                            )
                        )
            
            logger.info(f"Đã tải {len(self.legal_corpus)} văn bản pháp luật")
            return self.legal_corpus
            
        except Exception as e:
            logger.error(f"Lỗi khi tải corpus: {e}")
            raise
    
    def _create_sample_corpus(self) -> List[LegalDocument]:
        """Tạo corpus mẫu cho demo"""
        sample_docs = [
            {
                "law_id": "Luật Phòng, chống ma túy",
                "article_id": "32",
                "content": "Người nghiện ma túy từ đủ 18 tuổi trở lên bị áp dụng biện pháp xử lý hành chính đưa vào cơ sở cai nghiện bắt buộc theo quy định của Luật Xử lý vi phạm hành chính trong các trường hợp sau đây: a) Đã được áp dụng biện pháp giáo dục tại xã, phường, thị trấn mà tái phạm; b) Bị phát hiện sử dụng chất ma túy một cách trái phép trong thời gian cai nghiện ma túy tự nguyện tại gia đình, cộng đồng hoặc cơ sở cai nghiện ma túy tự nguyện."
            },
            {
                "law_id": "Luật Phòng, chống ma túy", 
                "article_id": "33",
                "content": "Thời hạn đưa vào cơ sở cai nghiện bắt buộc từ 12 tháng đến 24 tháng. Trường hợp sau khi hết thời hạn cai nghiện bắt buộc mà người được cai nghiện chưa bỏ được tệ nạn ma túy thì có thể gia hạn cai nghiện bắt buộc nhưng không quá 12 tháng."
            },
            {
                "law_id": "Bộ luật Dân sự",
                "article_id": "18", 
                "content": "Mọi người đều có quyền bất khả xâm phạm về thân thể. Không ai được xâm phạm thân thể của người khác dưới mọi hình thức."
            }
        ]
        
        return [LegalDocument(**doc) for doc in sample_docs]

class LLMGenerator:
    """LLM Generator cho câu trả lời"""
    
    def __init__(self, local_model_dir: Optional[str] = None):
        self.local_model_dir = local_model_dir
        self.local_model = None
        self.local_tokenizer = None
        self._try_load_local_model()
    
    def _try_load_local_model(self):
        """Tải model local nếu có"""
        if self.local_model_dir:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                logger.info(f"Đang tải model local từ {self.local_model_dir}...")
                
                self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_dir)
                if self.local_tokenizer.pad_token is None:
                    self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_dir,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    device_map='auto' if device == 'cuda' else None,
                    low_cpu_mem_usage=True
                )
                self.local_model.eval()
                logger.info("Đã tải model local thành công")
                
            except Exception as e:
                logger.error(f"Không thể load model local: {e}")
                self.local_model = None
                self.local_tokenizer = None
    
    def generate_answer(self, question: ALQACQuestion, 
                       retrieved_docs: List[RetrievalResult]) -> Dict[str, Any]:
        """Sinh câu trả lời"""
        if self.local_model and self.local_tokenizer:
            return self._generate_with_local_model(question, retrieved_docs)
        else:
            return self._generate_rule_based_answer(question, retrieved_docs)
    
    def _generate_with_local_model(self, question: ALQACQuestion, 
                                  retrieved_docs: List[RetrievalResult]) -> Dict[str, Any]:
        """Sinh câu trả lời với model local"""
        try:
            context = self._build_context(retrieved_docs)
            prompt = self._build_prompt(question, context)
            
            device = self.local_model.device
            inputs = self.local_tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
            
            response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()
            
            return {
                'answer': answer,
                'confidence': 0.8,
                'reasoning': 'Generated by local fine-tuned LLM',
                'retrieved_docs': len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi sinh câu trả lời từ local model: {e}")
            return self._generate_rule_based_answer(question, retrieved_docs)
    
    def _build_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Xây dựng context từ retrieved documents"""
        if not retrieved_docs:
            return "Không có tài liệu liên quan được tìm thấy."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            context_parts.append(
                f"Tài liệu {i} ({doc.document.law_id} - Điều {doc.document.article_id}):\n"
                f"{doc.document.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: ALQACQuestion, context: str) -> str:
        """Xây dựng prompt cho LLM"""
        base_prompt = f"""
Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi sau đây dựa trên các tài liệu pháp luật được cung cấp.

Câu hỏi ({question.question_type}): {question.text}

Tài liệu tham khảo:
{context}

Yêu cầu:
"""
        
        if question.question_type == "Đúng/Sai":
            base_prompt += """
- Trả lời chỉ "Đúng" hoặc "Sai"
"""
        elif question.question_type == "Trắc nghiệm":
            base_prompt += """
- Trả lời chỉ một trong các lựa chọn A, B, C, D
"""
        else:  # Tự luận
            base_prompt += """
- Trả lời chi tiết và đầy đủ
"""
        
        return base_prompt
    
    def _generate_rule_based_answer(self, question: ALQACQuestion, 
                                   retrieved_docs: List[RetrievalResult]) -> Dict[str, Any]:
        """Sinh câu trả lời dựa trên rule-based"""
        if not retrieved_docs:
            return {
                'answer': 'Không tìm thấy tài liệu liên quan',
                'confidence': 0.1,
                'reasoning': 'No relevant documents found',
                'retrieved_docs': 0
            }
        
        if question.question_type == "Đúng/Sai":
            top_doc = retrieved_docs[0]
            question_lower = question.text.lower()
            doc_lower = top_doc.document.content.lower()
            
            question_words = set(question_lower.split())
            doc_words = set(doc_lower.split())
            common_words = question_words.intersection(doc_words)
            
            confidence = len(common_words) / len(question_words) if question_words else 0
            
            if confidence > 0.5:
                answer = "Đúng"
            else:
                answer = "Sai"
            
            return {
                'answer': answer,
                'confidence': min(confidence, 0.7),
                'reasoning': f'Rule-based matching với {len(common_words)} từ khóa chung',
                'retrieved_docs': len(retrieved_docs)
            }
        
        return {
            'answer': 'Cần thêm thông tin để trả lời',
            'confidence': 0.3,
            'reasoning': 'Rule-based system limitation',
            'retrieved_docs': len(retrieved_docs)
        }

class ALQACEvaluator:
    """Lớp đánh giá hiệu suất cho ALQAC"""
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset tất cả metrics"""
        self.task1_metrics = {
            'precision': [],
            'recall': [],
            'f2': []
        }
        self.task2_metrics = {
            'accuracy': [],
            'correct_answers': 0,
            'total_questions': 0
        }
    
    def evaluate_task1(self, predicted_docs: List[str], 
                      ground_truth_docs: List[Dict[str, str]]) -> Dict[str, float]:
        """Đánh giá Task 1 (Document Retrieval)"""
        gt_ids = set()
        for doc in ground_truth_docs:
            doc_id = f"{doc['law_id']}_{doc['article_id']}"
            gt_ids.add(doc_id)
        
        predicted_set = set(predicted_docs)
        
        if predicted_set:
            precision = len(predicted_set.intersection(gt_ids)) / len(predicted_set)
        else:
            precision = 0.0
        
        if gt_ids:
            recall = len(predicted_set.intersection(gt_ids)) / len(gt_ids)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f2 = (5 * precision * recall) / (4 * precision + recall)
        else:
            f2 = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f2': f2
        }
        
        for key, value in metrics.items():
            self.task1_metrics[key].append(value)
        
        return metrics
    
    def evaluate_task2(self, predicted_answer: str, ground_truth_answer: str) -> Dict[str, float]:
        """Đánh giá Task 2 (Question Answering)"""
        pred_normalized = predicted_answer.strip().lower()
        gt_normalized = ground_truth_answer.strip().lower()
        
        is_correct = pred_normalized == gt_normalized
        
        self.task2_metrics['total_questions'] += 1
        if is_correct:
            self.task2_metrics['correct_answers'] += 1
        
        accuracy = self.task2_metrics['correct_answers'] / self.task2_metrics['total_questions']
        self.task2_metrics['accuracy'].append(1.0 if is_correct else 0.0)
        
        return {
            'correct': is_correct,
            'accuracy': accuracy
        }
    
    def get_summary_metrics(self) -> Dict[str, Dict[str, float]]:
        """Lấy tổng hợp metrics"""
        summary = {}
        
        if self.task1_metrics['precision']:
            summary['task1'] = {
                'avg_precision': np.mean(self.task1_metrics['precision']),
                'avg_recall': np.mean(self.task1_metrics['recall']),
                'avg_f2': np.mean(self.task1_metrics['f2']),
                'std_precision': np.std(self.task1_metrics['precision']),
                'std_recall': np.std(self.task1_metrics['recall']),
                'std_f2': np.std(self.task1_metrics['f2'])
            }
        
        if self.task2_metrics['total_questions'] > 0:
            summary['task2'] = {
                'accuracy': self.task2_metrics['correct_answers'] / self.task2_metrics['total_questions'],
                'correct_answers': self.task2_metrics['correct_answers'],
                'total_questions': self.task2_metrics['total_questions']
            }
        
        return summary

class BGE_M3ALQACSystem:
    """Hệ thống ALQAC sử dụng BGE-M3"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Khởi tạo các components
        self.data_loader = ALQACDataLoader()
        self.retriever = BGE_M3Retriever(
            model_name=self.config['bge_model'],
            device=self.config['device']
        )
        self.llm_generator = LLMGenerator(
            local_model_dir=self.config.get('llm_local_model_dir')
        )
        self.evaluator = ALQACEvaluator()
        
        # State
        self.is_indexed = False
        self.questions = []
        self.legal_corpus = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Cấu hình mặc định"""
        return {
            'bge_model': 'BAAI/bge-m3',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'top_k': 3,  # Giảm để tăng precision
            'llm_local_model_dir': './finetune_standard'
        }
    
    def load_data(self, questions_file: str, corpus_file: str = None):
        """Tải dữ liệu câu hỏi và corpus"""
        logger.info("Đang tải dữ liệu...")
        
        self.questions = self.data_loader.load_questions(questions_file)
        
        if corpus_file:
            self.legal_corpus = self.data_loader.load_legal_corpus(corpus_file)
        else:
            self.legal_corpus = self.data_loader.load_legal_corpus('alqac25_law.json')
        
        logger.info(f"Đã tải {len(self.questions)} câu hỏi và {len(self.legal_corpus)} tài liệu")
    
    def build_index(self):
        """Xây dựng index cho hệ thống"""
        if not self.legal_corpus:
            raise ValueError("Chưa tải corpus pháp luật")
        
        logger.info("Đang xây dựng BGE-M3 index...")
        self.retriever.build_index(self.legal_corpus)
        self.is_indexed = True
        logger.info("Hoàn thành xây dựng index")
    
    def process_single_question(self, question: ALQACQuestion) -> Dict[str, Any]:
        """Xử lý một câu hỏi"""
        if not self.is_indexed:
            raise ValueError("Chưa xây dựng index")
        
        start_time = time.time()
        
        # BGE-M3 Retrieval
        logger.info(f"Xử lý câu hỏi {question.question_id} - BGE-M3 Retrieval")
        retrieved_docs = self.retriever.retrieve(
            question.text, 
            top_k=self.config['top_k']
        )
        
        # LLM Generation
        logger.info(f"LLM Generation")
        answer_result = self.llm_generator.generate_answer(question, retrieved_docs)
        
        processing_time = time.time() - start_time
        
        # Đánh giá kết quả
        predicted_doc_ids = [result.document_id for result in retrieved_docs]
        task1_metrics = self.evaluator.evaluate_task1(
            predicted_doc_ids, 
            question.relevant_articles
        )
        
        task2_metrics = self.evaluator.evaluate_task2(
            answer_result['answer'],
            question.answer
        )
        
        return {
            'question_id': question.question_id,
            'question_type': question.question_type,
            'question_text': question.text,
            'predicted_answer': answer_result['answer'],
            'ground_truth_answer': question.answer,
            'retrieved_documents': [
                {
                    'doc_id': result.document_id,
                    'law_id': result.document.law_id,
                    'article_id': result.document.article_id,
                    'score': result.score
                }
                for result in retrieved_docs
            ],
            'relevant_articles': question.relevant_articles,
            'task1_metrics': task1_metrics,
            'task2_metrics': task2_metrics,
            'processing_time': processing_time,
            'answer_confidence': answer_result.get('confidence', 0.0)
        }
    
    def process_all_questions(self) -> List[Dict[str, Any]]:
        """Xử lý tất cả câu hỏi"""
        if not self.questions:
            raise ValueError("Chưa tải câu hỏi")
        
        logger.info(f"Bắt đầu xử lý {len(self.questions)} câu hỏi...")
        
        results = []
        for i, question in enumerate(self.questions, 1):
            logger.info(f"Đang xử lý câu hỏi {i}/{len(self.questions)}")
            
            try:
                result = self.process_single_question(question)
                results.append(result)
                
                if i % 10 == 0:
                    logger.info(f"Đã hoàn thành {i} câu hỏi")
                    
            except Exception as e:
                logger.error(f"Lỗi khi xử lý câu hỏi {question.question_id}: {e}")
                continue
        
        logger.info("Hoàn thành xử lý tất cả câu hỏi")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Lưu kết quả ra file"""
        try:
            summary_metrics = self.evaluator.get_summary_metrics()
            
            output_data = {
                'summary_metrics': summary_metrics,
                'total_questions': len(results),
                'results': results,
                'config': self.config,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Đã lưu kết quả vào {output_file}")
            self._print_summary(summary_metrics)
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả: {e}")
    
    def _print_summary(self, metrics: Dict[str, Dict[str, float]]):
        """In tóm tắt kết quả"""
        print("\n" + "="*50)
        print("TỔNG HỢP KẾT QUẢ ALQAC 2025 - BGE-M3")
        print("="*50)
        
        if 'task1' in metrics:
            print("\nTask 1 - Legal Document Retrieval:")
            print(f"  Precision: {metrics['task1']['avg_precision']:.4f}")
            print(f"  Recall:    {metrics['task1']['avg_recall']:.4f}")
            print(f"  F2-score:  {metrics['task1']['avg_f2']:.4f}")
        
        if 'task2' in metrics:
            print(f"\nTask 2 - Legal Question Answering:")
            print(f"  Accuracy:  {metrics['task2']['accuracy']:.4f}")
            print(f"  Correct:   {metrics['task2']['correct_answers']}/{metrics['task2']['total_questions']}")
        
        print("="*50)

def main():
    """Hàm main để chạy hệ thống BGE-M3"""
    # Cấu hình tối ưu cho precision
    config = {
        'bge_model': 'BAAI/bge-m3',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'top_k': 3,  # Giảm để tăng precision
        'llm_local_model_dir': './finetune_standard'
    }
    
    # Khởi tạo hệ thống
    system = BGE_M3ALQACSystem(config)
    
    # Sử dụng file test
    training_file = 'alqac25_test_split.json'
    
    try:
        # Tải dữ liệu
        system.load_data(training_file)
        
        # Xây dựng index
        system.build_index()
        
        # Xử lý câu hỏi
        results = system.process_all_questions()
        
        # Lưu kết quả
        system.save_results(results, 'bge_m3_alqac_results.json')
        
        print("\nHệ thống BGE-M3 đã chạy thành công!")
        print("Kết quả đã được lưu vào 'bge_m3_alqac_results.json'")
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy hệ thống: {e}")
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main() 