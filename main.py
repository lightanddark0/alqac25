# multi_stage_retrieval_alqac.py

import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from collections import defaultdict, Counter
import re

# Core libraries
try:
    from rank_bm25 import BM25Okapi
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    from underthesea import word_tokenize
    import requests
except ImportError as e:
    print(f"Thiếu thư viện: {e}")
    print("Chạy: pip install rank-bm25 transformers torch scikit-learn underthesea requests")
    exit(1)

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
    stage_scores: Dict[str, float]

class ALQACDataLoader:
    """Lớp để tải và xử lý dữ liệu ALQAC"""
    
    def __init__(self):
        self.questions: List[ALQACQuestion] = []
        self.legal_corpus: List[LegalDocument] = []
        
    def load_questions(self, json_file_path: str) -> List[ALQACQuestion]:
        """
        Tải câu hỏi từ file JSON
        
        Args:
            json_file_path: Đường dẫn đến file JSON chứa câu hỏi
            
        Returns:
            Danh sách câu hỏi đã được parse
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Xử lý trường hợp file chứa một câu hỏi hoặc danh sách câu hỏi
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
        """
        Tải corpus văn bản pháp luật
        
        Args:
            corpus_path: Đường dẫn đến file chứa corpus pháp luật
            
        Returns:
            Danh sách văn bản pháp luật
        """
        try:
            # Tạo corpus mẫu nếu file không tồn tại
            if not Path(corpus_path).exists():
                logger.warning(f"File corpus không tồn tại: {corpus_path}")
                logger.info("Tạo corpus mẫu...")
                self.legal_corpus = self._create_sample_corpus()
            else:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    corpus_data = json.load(f)
                # Định dạng mới: mỗi luật có id và articles (list), mỗi article có id và text
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

class VietnameseTextProcessor:
    """Lớp xử lý văn bản tiếng Việt chuyên biệt"""
    
    def __init__(self):
        # Stopwords tiếng Việt được tối ưu cho domain pháp luật
        self.legal_stopwords = {
            'là', 'của', 'và', 'có', 'được', 'theo', 'trong', 'với', 'từ', 'đến',
            'về', 'cho', 'khi', 'nếu', 'để', 'như', 'bị', 'do', 'tại', 'trên',
            'dưới', 'qua', 'sau', 'trước', 'giữa', 'trong', 'ngoài', 'cùng',
            'cả', 'mọi', 'các', 'những', 'này', 'đó', 'ấy', 'kia', 'đây', 'đấy'
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Tiền xử lý văn bản tiếng Việt
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Danh sách token đã được xử lý
        """
        if not text:
            return []
        
        # Chuẩn hóa văn bản
        text = text.lower().strip()
        
        # Loại bỏ ký tự đặc biệt nhưng giữ lại dấu câu quan trọng
        text = re.sub(r'[^\w\s.,;:!?()]', ' ', text)
        
        # Word segmentation cho tiếng Việt
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback nếu underthesea gặp lỗi
            tokens = text.split()
        
        # Lọc stopwords và token rỗng
        processed_tokens = [
            token for token in tokens 
            if token and len(token) > 1 and token not in self.legal_stopwords
        ]
        
        return processed_tokens

class Stage1_BM25Retriever:
    """Giai đoạn 1: BM25 Pre-ranking với tối ưu cho tiếng Việt"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.text_processor = VietnameseTextProcessor()
        self.bm25 = None
        self.documents = []
        self.processed_docs = []
        
    def index_documents(self, documents: List[LegalDocument]):
        """Lập chỉ mục cho tài liệu"""
        logger.info("Đang lập chỉ mục BM25...")
        
        self.documents = documents
        self.processed_docs = []
        
        for doc in documents:
            processed_content = self.text_processor.preprocess_text(doc.content)
            self.processed_docs.append(processed_content)
        
        self.bm25 = BM25Okapi(self.processed_docs, k1=self.k1, b=self.b)
        logger.info(f"Đã lập chỉ mục {len(documents)} tài liệu")
    
    def retrieve(self, query: str, top_k: int = 100) -> List[RetrievalResult]:
        """
        Truy xuất tài liệu sử dụng BM25
        
        Args:
            query: Câu truy vấn
            top_k: Số lượng tài liệu trả về
            
        Returns:
            Danh sách kết quả được sắp xếp theo điểm BM25
        """
        if not self.bm25:
            raise ValueError("Chưa lập chỉ mục tài liệu")
        
        # Tiền xử lý query
        processed_query = self.text_processor.preprocess_text(query)
        
        # Tính điểm BM25
        scores = self.bm25.get_scores(processed_query)
        
        # Tạo kết quả
        results = []
        for i, score in enumerate(scores):
            if i < len(self.documents):
                result = RetrievalResult(
                    document_id=self.documents[i].get_id(),
                    score=float(score),
                    document=self.documents[i],
                    stage_scores={'bm25': float(score)}
                )
                results.append(result)
        
        # Sắp xếp và trả về top-k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

class Stage2_BERTReranker:
    """Giai đoạn 2: BERT Re-ranking sử dụng PhoBERT"""
    
    def __init__(self, model_name: str = "vinai/phobert-base", device: str = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            logger.info(f"Đang tải PhoBERT từ {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Đã tải PhoBERT thành công")
        except Exception as e:
            logger.error(f"Lỗi khi tải PhoBERT: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode văn bản thành vector embedding
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Vector embedding
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                max_length=256, 
                truncation=True, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Sử dụng [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Lỗi khi encode text: {e}")
            # Trả về vector ngẫu nhiên nếu có lỗi
            return np.random.rand(768)
    
    def rerank(self, query: str, candidates: List[RetrievalResult], 
              top_k: int = 20, alpha: float = 0.3) -> List[RetrievalResult]:
        """
        Re-rank candidates sử dụng BERT
        
        Args:
            query: Câu truy vấn
            candidates: Danh sách candidates từ stage 1
            top_k: Số lượng kết quả trả về
            alpha: Trọng số kết hợp BM25 và BERT (0-1)
            
        Returns:
            Danh sách kết quả đã được re-rank
        """
        if not candidates:
            return []
        
        logger.info(f"Đang re-rank {len(candidates)} candidates với PhoBERT...")
        
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Encode documents và tính similarity
        for candidate in candidates:
            doc_embedding = self.encode_text(candidate.document.content)
            
            # Tính cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
            
            # Kết hợp điểm BM25 và BERT
            bm25_score = candidate.stage_scores.get('bm25', 0)
            combined_score = alpha * bm25_score + (1 - alpha) * similarity
            
            # Cập nhật scores
            candidate.stage_scores['bert'] = float(similarity)
            candidate.stage_scores['combined'] = float(combined_score)
            candidate.score = float(combined_score)
        
        # Sắp xếp theo điểm kết hợp
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        logger.info("Hoàn thành re-ranking")
        return candidates[:top_k]

class Stage3_LLMGenerator:
    """Giai đoạn 3: LLM Generation với prompting techniques"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", local_model_dir: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.has_api = api_key is not None
        self.local_model_dir = local_model_dir
        self.local_model = None
        self.local_tokenizer = None
        self._try_load_local_model()
    
    def _try_load_local_model(self):
        if self.local_model_dir:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                logger.info(f"Đang tải model local từ {self.local_model_dir}...")
                
                # Tải tokenizer
                self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_dir)
                if self.local_tokenizer.pad_token is None:
                    self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
                
                # Tải model với cấu hình phù hợp với hardware
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Sử dụng device: {device}")
                
                if device == 'cpu':
                    # Sử dụng CPU với cấu hình nhẹ
                    self.local_model = AutoModelForCausalLM.from_pretrained(
                        self.local_model_dir,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        device_map='auto' if device == 'cuda' else None
                    )
                else:
                    # Sử dụng GPU
                    self.local_model = AutoModelForCausalLM.from_pretrained(
                        self.local_model_dir,
                        torch_dtype=torch.float16,
                        device_map='auto'
                    )
                
                self.local_model.eval()
                logger.info("Đã tải model local thành công")
                
            except Exception as e:
                logger.error(f"Không thể load model local từ {self.local_model_dir}: {e}")
                logger.info("Sẽ sử dụng rule-based fallback")
                self.local_model = None
                self.local_tokenizer = None
    
    def generate_answer(self, question: ALQACQuestion, 
                       retrieved_docs: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Sinh câu trả lời sử dụng LLM hoặc model local
        """
        # Nếu có model local đã fine-tune thì ưu tiên dùng
        if self.local_model and self.local_tokenizer:
            print("model local")
            prompt = self._build_prompt(question, self._build_context(retrieved_docs))
            try:
                import torch
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
                # Loại bỏ prompt khỏi response nếu có
                answer = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()
                return {
                    'answer': answer,
                    'confidence': 0.8,
                    'reasoning': 'Generated by local fine-tuned LLM',
                    'retrieved_docs': len(retrieved_docs)
                }
            except Exception as e:
                logger.error(f"Lỗi khi sinh câu trả lời từ local model: {e}")
                # fallback sang rule-based
                return self._generate_rule_based_answer(question, retrieved_docs)
        
        # Nếu có API thì dùng API
        if self.has_api:

            print("api")
            # Tạo context từ retrieved documents
            context = self._build_context(retrieved_docs)
            
            # Tạo prompt dựa trên loại câu hỏi
            prompt = self._build_prompt(question, context)
            
            # Gọi API LLM (placeholder - cần implement API call)
            try:
                response = self._call_llm_api(prompt)
                return {
                    'answer': response,
                    'confidence': 0.8,
                    'reasoning': 'Generated by LLM',
                    'retrieved_docs': len(retrieved_docs)
                }
            except Exception as e:
                logger.error(f"Lỗi khi gọi LLM API: {e}")
                return self._generate_rule_based_answer(question, retrieved_docs)
        
        # Fallback rule-based
        return self._generate_rule_based_answer(question, retrieved_docs)
    
    def _build_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Xây dựng context từ các tài liệu được truy xuất"""
        if not retrieved_docs:
            return "Không có tài liệu liên quan được tìm thấy."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):  # Chỉ lấy top 5
            context_parts.append(
                f"Tài liệu {i} ({doc.document.law_id} - Điều {doc.document.article_id}):\n"
                f"{doc.document.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, question: ALQACQuestion, context: str) -> str:
        if question.question_type == "Đúng/Sai":
            prompt = f"""Dựa nội dung sau đây, hãy xác định câu sau là Đúng hay Sai.\n\n
                Lưu ý chỉ cần trả lời Đúng hay Sai không cần giải thích gì thêm.\n\n
                Câu hỏi: {question}\n\n
                {context}\n\n
                Đáp án: """
        if question.question_type == "Trắc nghiệm":
            prompt = f"""Dựa vào nội dung sau đây, hãy chọn đáp án đúng.\n\n
                Chỉ cần trả lời bằng một trong các lựa chọn A, B, C, hoặc D không
                cần giải thích gì thêm.\n\n
                Câu hỏi: {question}\n\n
                {context}\n\n
                Đáp án: """
        if question.question_type == "Tự luận":
            prompt = f"""Dựa nội dung sau đây, hãy trả lời ngắn gọn\n\n
                Câu hỏi: {question}\n\n
                {context}\n\n
                Đáp án: """
        return prompt
        
    def _generate_rule_based_answer(self, question: ALQACQuestion, 
                                   retrieved_docs: List[RetrievalResult]) -> Dict[str, Any]:
        """Sinh câu trả lời dựa trên rule-based (fallback)"""
        if not retrieved_docs:
            return {
                'answer': 'Không tìm thấy tài liệu liên quan',
                'confidence': 0.1,
                'reasoning': 'No relevant documents found',
                'retrieved_docs': 0
            }
        
        # Rule-based logic đơn giản cho câu hỏi Đúng/Sai
        if question.question_type == "Đúng/Sai":
            # Kiểm tra từ khóa trong tài liệu có điểm cao nhất
            top_doc = retrieved_docs[0]
            question_lower = question.text.lower()
            doc_lower = top_doc.document.content.lower()
            
            # Đếm từ khóa chung
            question_words = set(question_lower.split())
            doc_words = set(doc_lower.split())
            common_words = question_words.intersection(doc_words)
            
            confidence = len(common_words) / len(question_words) if question_words else 0
            
            # Heuristic đơn giản
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
    
    def _call_llm_api(self, prompt: str) -> str:
        """Gọi API LLM (placeholder)"""
        # Đây là placeholder - cần implement API call thực tế
        # Ví dụ với OpenAI API:
        # response = openai.ChatCompletion.create(...)
        return "Placeholder response"

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
        """
        Đánh giá Task 1 (Document Retrieval)
        
        Args:
            predicted_docs: Danh sách ID tài liệu được dự đoán
            ground_truth_docs: Danh sách tài liệu ground truth
            
        Returns:
            Dictionary chứa precision, recall, F2-measure
        """
        # Chuyển đổi ground truth thành set của IDs
        gt_ids = set()
        for doc in ground_truth_docs:
            doc_id = f"{doc['law_id']}_{doc['article_id']}"
            gt_ids.add(doc_id)
        
        predicted_set = set(predicted_docs)
        
        # Tính precision, recall
        if predicted_set:
            precision = len(predicted_set.intersection(gt_ids)) / len(predicted_set)
        else:
            precision = 0.0
        
        if gt_ids:
            recall = len(predicted_set.intersection(gt_ids)) / len(gt_ids)
        else:
            recall = 0.0
        
        # Tính F2-measure (ưu tiên recall)
        if precision + recall > 0:
            f2 = (5 * precision * recall) / (4 * precision + recall)
        else:
            f2 = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f2': f2
        }
        
        # Lưu metrics
        for key, value in metrics.items():
            self.task1_metrics[key].append(value)
        
        return metrics
    
    def evaluate_task2(self, predicted_answer: str, ground_truth_answer: str) -> Dict[str, float]:
        """
        Đánh giá Task 2 (Question Answering)
        
        Args:
            predicted_answer: Câu trả lời dự đoán
            ground_truth_answer: Câu trả lời đúng
            
        Returns:
            Dictionary chứa accuracy
        """
        # Chuẩn hóa câu trả lời
        pred_normalized = predicted_answer.strip().lower()
        gt_normalized = ground_truth_answer.strip().lower()
        
        is_correct = pred_normalized == gt_normalized
        
        self.task2_metrics['total_questions'] += 1
        if is_correct:
            self.task2_metrics['correct_answers'] += 1
        
        # Tính accuracy hiện tại
        accuracy = self.task2_metrics['correct_answers'] / self.task2_metrics['total_questions']
        self.task2_metrics['accuracy'].append(1.0 if is_correct else 0.0)
        
        return {
            'correct': is_correct,
            'accuracy': accuracy
        }
    
    def get_summary_metrics(self) -> Dict[str, Dict[str, float]]:
        """Lấy tổng hợp metrics"""
        summary = {}
        
        # Task 1 metrics
        if self.task1_metrics['precision']:
            summary['task1'] = {
                'avg_precision': np.mean(self.task1_metrics['precision']),
                'avg_recall': np.mean(self.task1_metrics['recall']),
                'avg_f2': np.mean(self.task1_metrics['f2']),
                'std_precision': np.std(self.task1_metrics['precision']),
                'std_recall': np.std(self.task1_metrics['recall']),
                'std_f2': np.std(self.task1_metrics['f2'])
            }
        
        # Task 2 metrics
        if self.task2_metrics['total_questions'] > 0:
            summary['task2'] = {
                'accuracy': self.task2_metrics['correct_answers'] / self.task2_metrics['total_questions'],
                'correct_answers': self.task2_metrics['correct_answers'],
                'total_questions': self.task2_metrics['total_questions']
            }
        
        return summary

class MultiStageRetrievalSystem:
    """Hệ thống Multi-stage Retrieval chính cho ALQAC 2025"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Khởi tạo các components
        self.data_loader = ALQACDataLoader()
        self.stage1 = Stage1_BM25Retriever(
            k1=self.config['bm25_k1'],
            b=self.config['bm25_b']
        )
        self.stage2 = Stage2_BERTReranker(
            model_name=self.config['bert_model'],
            device=self.config['device']
        )
        self.stage3 = Stage3_LLMGenerator(
            api_key=self.config.get('llm_api_key'),
            model=self.config.get('llm_model', 'llama3.2-7b-instruct'),
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
            'bm25_k1': 1.2,
            'bm25_b': 0.75,
            'bm25_top_k': 10,
            'bert_model': 'vinai/phobert-base',
            'bert_top_k': 3,
            'bert_alpha': 0.4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'llm_api_key': None,
            'llm_model': 'gpt-3.5-turbo',
            'llm_local_model_dir': None
        }
    
    def load_data(self, questions_file: str, corpus_file: str = None):
        """
        Tải dữ liệu câu hỏi và corpus
        
        Args:
            questions_file: File JSON chứa câu hỏi
            corpus_file: File chứa corpus pháp luật (optional)
        """
        logger.info("Đang tải dữ liệu...")
        
        # Tải câu hỏi
        self.questions = self.data_loader.load_questions(questions_file)
        
        # Tải corpus
        if corpus_file:
            self.legal_corpus = self.data_loader.load_legal_corpus(corpus_file)
        else:
            # Sử dụng file corpus mặc định 'alqac25_law.json' nếu không truyền vào
            self.legal_corpus = self.data_loader.load_legal_corpus('alqac25_law.json')
        
        logger.info(f"Đã tải {len(self.questions)} câu hỏi và {len(self.legal_corpus)} tài liệu")
    
    def build_index(self):
        """Xây dựng index cho hệ thống"""
        if not self.legal_corpus:
            raise ValueError("Chưa tải corpus pháp luật")
        
        logger.info("Đang xây dựng index...")
        self.stage1.index_documents(self.legal_corpus)
        self.is_indexed = True
        logger.info("Hoàn thành xây dựng index")
    
    def process_single_question(self, question: ALQACQuestion) -> Dict[str, Any]:
        """
        Xử lý một câu hỏi qua 3 stages
        
        Args:
            question: Câu hỏi ALQAC
            
        Returns:
            Kết quả xử lý đầy đủ
        """
        if not self.is_indexed:
            raise ValueError("Chưa xây dựng index")
        
        start_time = time.time()
        
        # Stage 1: BM25 Retrieval
        logger.info(f"Xử lý câu hỏi {question.question_id} - Stage 1: BM25")
        stage1_results = self.stage1.retrieve(
            question.text, 
            top_k=self.config['bm25_top_k']
        )
        
        # Stage 2: BERT Re-ranking  
        logger.info(f"Stage 2: BERT Re-ranking")
        stage2_results = self.stage2.rerank(
            question.text,
            stage1_results,
            top_k=self.config['bert_top_k'],
            alpha=self.config['bert_alpha']
        )
        
        # Stage 3: LLM Generation
        logger.info(f"Stage 3: LLM Generation")
        answer_result = self.stage3.generate_answer(question, stage2_results)
        
        processing_time = time.time() - start_time
        
        # Đánh giá kết quả
        predicted_doc_ids = [result.document_id for result in stage2_results]
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
                    'score': result.score,
                    'stage_scores': result.stage_scores
                }
                for result in stage2_results
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
                
                # Log tiến độ
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
            # Thêm summary metrics
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
            
            # In summary metrics
            self._print_summary(summary_metrics)
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả: {e}")
    
    def _print_summary(self, metrics: Dict[str, Dict[str, float]]):
        """In tóm tắt kết quả"""
        print("\n" + "="*50)
        print("TỔNG HỢP KẾT QUẢ ALQAC 2025")
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
    """Hàm main để chạy hệ thống"""
    # Cấu hình
    config = {
        'bm25_k1': 2,
        'bm25_b': 0.75,
        'bm25_top_k': 5,  # Giảm để phù hợp với máy yếu
        'bert_model': 'vinai/phobert-base',
        'bert_top_k': 5,  # Giảm để phù hợp với máy yếu
        'bert_alpha': 0.3,
        'device': 'cpu',  # Dùng CPU cho máy yếu
        'llm_api_key': None,  # Không dùng LLM API
        'llm_model': 'llama3.2-7b-instruct',
        'llm_local_model_dir': './finetune_standard'  # Đường dẫn model đã fine-tune
    }
    
    # Khởi tạo hệ thống
    system = MultiStageRetrievalSystem(config)
    
    # Sử dụng file training mặc định 'alqac25_train.json'
    training_file = 'alqac25_test_split.json'
    
    try:
        # Tải dữ liệu
        system.load_data(training_file)
        
        # Xây dựng index
        system.build_index()
        
        # Xử lý câu hỏi
        results = system.process_all_questions()
        
        # Lưu kết quả
        system.save_results(results, 'alqac_results.json')
        
        print("\nHệ thống đã chạy thành công!")
        print("Kết quả đã được lưu vào 'alqac_results.json'")
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy hệ thống: {e}")
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
