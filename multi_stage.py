
"""
Multi-stage Retrieval System for ALQAC 2025
Hệ thống truy xuất văn bản pháp luật tiếng Việt theo phương pháp 3 giai đoạn

Giai đoạn 1: BM25 pre-ranking và thu hẹp phạm vi tìm kiếm
Giai đoạn 2: BERT-based re-ranking để cải thiện độ chính xác
Giai đoạn 3: LLM prompting techniques cho câu trả lời cuối cùng

Author: AI Assistant
Date: 2025
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Any
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

# Thư viện cho BM25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Cần cài đặt: pip install rank-bm25")

# Thư viện cho BERT/PhoBERT
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
except ImportError:
    print("Cần cài đặt: pip install transformers sentence-transformers torch")

# Thư viện cho preprocessing tiếng Việt
try:
    import underthesea
except ImportError:
    print("Cần cài đặt: pip install underthesea")

# Thư viện cho LLM API (ví dụ: OpenAI compatible)
try:
    from openai import OpenAI
except ImportError:
    print("Cần cài đặt: pip install openai")

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Class đại diện cho một văn bản pháp luật"""
    id: str
    title: str
    content: str
    article_number: str = ""
    law_name: str = ""

@dataclass
class RetrievalResult:
    """Class đại diện cho kết quả truy xuất"""
    document: Document
    score: float
    stage: str

class VietnameseTextProcessor:
    """Xử lý văn bản tiếng Việt"""

    def __init__(self):
        self.stopwords = set([
            'là', 'của', 'và', 'có', 'được', 'trong', 'cho', 'với', 'các', 'một',
            'này', 'đó', 'để', 'từ', 'theo', 'về', 'tại', 'trên', 'dưới', 'qua'
        ])

    def word_segment(self, text: str) -> List[str]:
        """Tách từ tiếng Việt"""
        try:
            return underthesea.word_tokenize(text, format="text").split()
        except:
            # Fallback nếu underthesea không có
            return text.split()

    def preprocess_text(self, text: str) -> List[str]:
        """Tiền xử lý văn bản: tách từ, loại bỏ stopwords, lowercase"""
        # Tách từ
        words = self.word_segment(text.lower())

        # Loại bỏ stopwords và các token không mong muốn
        processed_words = []
        for word in words:
            if (word not in self.stopwords and 
                len(word) > 1 and 
                word.isalpha()):
                processed_words.append(word)

        return processed_words

class Stage1_BM25Retriever:
    """Giai đoạn 1: BM25 Retrieval"""

    def __init__(self, documents: List[Document], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.text_processor = VietnameseTextProcessor()

        # Tiền xử lý corpus
        logger.info("Đang tiền xử lý corpus cho BM25...")
        self.processed_corpus = []
        for doc in documents:
            # Kết hợp title và content để tăng khả năng truy xuất
            full_text = f"{doc.title} {doc.content}"
            processed_text = self.text_processor.preprocess_text(full_text)
            self.processed_corpus.append(processed_text)

        # Khởi tạo BM25
        self.bm25 = BM25Okapi(self.processed_corpus, k1=k1, b=b)
        logger.info(f"Đã khởi tạo BM25 với {len(documents)} văn bản")

    def search(self, query: str, top_k: int = 100) -> List[RetrievalResult]:
        """Tìm kiếm BM25"""
        # Tiền xử lý query
        processed_query = self.text_processor.preprocess_text(query)

        # Tính điểm BM25
        scores = self.bm25.get_scores(processed_query)

        # Lấy top_k kết quả
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Chỉ lấy những kết quả có điểm > 0
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    stage="BM25"
                ))

        logger.info(f"BM25 tìm thấy {len(results)} văn bản liên quan")
        return results

class Stage2_BERTReranker:
    """Giai đoạn 2: BERT-based Re-ranking"""

    def __init__(self, model_name: str = "vinai/phobert-base"):
        self.model_name = model_name

        # Khởi tạo tokenizer và model
        logger.info(f"Đang tải mô hình {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Sử dụng GPU nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Đã tải mô hình BERT trên {self.device}")

    def encode_text(self, text: str, max_length: int = 256) -> torch.Tensor:
        """Mã hóa văn bản thành vector"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )

        # Chuyển sang device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Sử dụng CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def compute_similarity(self, query: str, document: Document) -> float:
        """Tính độ tương đồng giữa query và document"""
        # Tạo text pairs cho classification
        doc_text = f"{document.title} {document.content}"

        # Encode query và document
        query_embedding = self.encode_text(query)
        doc_embedding = self.encode_text(doc_text)

        # Tính cosine similarity
        similarity = F.cosine_similarity(query_embedding, doc_embedding)

        return float(similarity.cpu().item())

    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 20) -> List[RetrievalResult]:
        """Re-rank các candidates sử dụng BERT"""
        logger.info(f"Đang re-rank {len(candidates)} văn bản với BERT...")

        reranked_results = []

        for result in candidates:
            # Tính điểm similarity mới
            bert_score = self.compute_similarity(query, result.document)

            # Kết hợp điểm BM25 và BERT (có thể điều chỉnh trọng số)
            combined_score = 0.3 * result.score + 0.7 * bert_score

            reranked_results.append(RetrievalResult(
                document=result.document,
                score=combined_score,
                stage="BERT_Rerank"
            ))

        # Sắp xếp lại theo điểm mới
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Hoàn thành re-ranking, trả về top {top_k}")
        return reranked_results[:top_k]

class Stage3_LLMGenerator:
    """Giai đoạn 3: LLM with Prompting Techniques"""

    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name

        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            logger.warning("Không có API key, chỉ trả về template prompt")
            self.client = None

    def create_legal_prompt(self, question: str, retrieved_docs: List[RetrievalResult]) -> str:
        """Tạo prompt cho LLM với kỹ thuật prompting"""

        # Chuẩn bị context từ các văn bản được truy xuất
        context_parts = []
        for i, result in enumerate(retrieved_docs[:5], 1):  # Chỉ lấy top 5
            doc = result.document
            context_parts.append(f"""
Văn bản {i}:
Tiêu đề: {doc.title}
Điều luật: {doc.article_number}
Nội dung: {doc.content[:500]}...
""")

        context = "\n".join(context_parts)

        # Prompt engineering cho legal QA
        prompt = f"""Bạn là một chuyên gia pháp luật Việt Nam với kinh nghiệm sâu rộng về hệ thống pháp luật.

NHIỆM VỤ:
Dựa vào các văn bản pháp luật được cung cấp bên dưới, hãy trả lời câu hỏi một cách chính xác và chi tiết.

QUY TẮC QUAN TRỌNG:
1. CHỈ sử dụng thông tin từ các văn bản pháp luật được cung cấp
2. KHÔNG bịa đặt hoặc suy luận ngoài phạm vi các văn bản đã cho
3. Nếu không đủ thông tin để trả lời, hãy nói rõ điều đó
4. Trích dẫn cụ thể điều luật và văn bản liên quan
5. Giải thích rõ ràng, dễ hiểu cho người không chuyên

CÂU HỎI:
{question}

CÁC VĂN BẢN PHÁP LUẬT LIÊN QUAN:
{context}

HƯỚNG DẪN TRẢ LỜI:
- Bắt đầu bằng việc xác định các điều luật áp dụng
- Giải thích nội dung các điều luật có liên quan
- Đưa ra câu trả lời cụ thể cho câu hỏi
- Kết luận ngắn gọn

TRẢ LỜI:"""

        return prompt

    def generate_answer(self, question: str, retrieved_docs: List[RetrievalResult]) -> str:
        """Sinh câu trả lời sử dụng LLM"""

        prompt = self.create_legal_prompt(question, retrieved_docs)

        if not self.client:
            return f"Template prompt đã được tạo:\n\n{prompt}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Giảm creativity để tăng tính chính xác
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Lỗi khi gọi LLM API: {e}")
            return f"Lỗi API. Template prompt:\n\n{prompt}"

class MultiStageRetrievalSystem:
    """Hệ thống Multi-stage Retrieval hoàn chỉnh"""

    def __init__(self, documents: List[Document], llm_api_key: str = None):
        self.documents = documents

        # Khởi tạo các giai đoạn
        logger.info("Khởi tạo hệ thống Multi-stage Retrieval...")

        # Stage 1: BM25
        self.stage1 = Stage1_BM25Retriever(documents)

        # Stage 2: BERT Re-ranker
        self.stage2 = Stage2_BERTReranker()

        # Stage 3: LLM Generator
        self.stage3 = Stage3_LLMGenerator(api_key=llm_api_key)

        logger.info("Đã khởi tạo xong hệ thống!")

    def search_and_answer(self, question: str, bm25_top_k: int = 100, bert_top_k: int = 20) -> Dict[str, Any]:
        """Tìm kiếm và trả lời câu hỏi qua 3 giai đoạn"""

        logger.info(f"Bắt đầu xử lý câu hỏi: {question}")

        # Stage 1: BM25 Retrieval
        stage1_results = self.stage1.search(question, top_k=bm25_top_k)

        if not stage1_results:
            return {
                "question": question,
                "answer": "Không tìm thấy văn bản pháp luật liên quan đến câu hỏi.",
                "retrieved_documents": [],
                "stages": {
                    "stage1_count": 0,
                    "stage2_count": 0,
                    "stage3_used": False
                }
            }

        # Stage 2: BERT Re-ranking
        stage2_results = self.stage2.rerank(question, stage1_results, top_k=bert_top_k)

        # Stage 3: LLM Answer Generation
        answer = self.stage3.generate_answer(question, stage2_results)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": result.document.id,
                    "title": result.document.title,
                    "score": result.score,
                    "stage": result.stage
                }
                for result in stage2_results
            ],
            "stages": {
                "stage1_count": len(stage1_results),
                "stage2_count": len(stage2_results),
                "stage3_used": True
            }
        }

    def save_system(self, filepath: str):
        """Lưu hệ thống để sử dụng lại"""
        system_data = {
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "article_number": doc.article_number,
                    "law_name": doc.law_name
                }
                for doc in self.documents
            ],
            "bm25_corpus": self.stage1.processed_corpus
        }

        with open(filepath, "wb") as f:
            pickle.dump(system_data, f)

        logger.info(f"Đã lưu hệ thống vào {filepath}")

    @classmethod
    def load_system(cls, filepath: str, llm_api_key: str = None):
        """Tải hệ thống đã lưu"""
        with open(filepath, "rb") as f:
            system_data = pickle.load(f)

        # Tái tạo documents
        documents = []
        for doc_data in system_data["documents"]:
            documents.append(Document(**doc_data))

        return cls(documents, llm_api_key)

def load_legal_documents_from_json(filepath: str) -> List[Document]:
    """Tải dữ liệu từ file JSON"""
    documents = []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

        for item in data:
            doc = Document(
                id=item.get("id", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                article_number=item.get("article_number", ""),
                law_name=item.get("law_name", "")
            )
            documents.append(doc)

    logger.info(f"Đã tải {len(documents)} văn bản từ {filepath}")
    return documents

def main():
    """Hàm main để demo hệ thống"""

    # Tạo dữ liệu mẫu nếu không có file JSON
    sample_documents = [
        Document(
            id="1",
            title="Bộ luật Dân sự 2015 - Điều 1",
            content="Bộ luật này quy định về quan hệ dân sự; cá nhân, pháp nhân và các chủ thể khác tham gia quan hệ dân sự; căn cứ phát sinh, thay đổi, chấm dứt quan hệ dân sự.",
            article_number="Điều 1",
            law_name="Bộ luật Dân sự 2015"
        ),
        Document(
            id="2", 
            title="Bộ luật Dân sự 2015 - Điều 18",
            content="Cá nhân có quyền, nghĩa vụ dân sự từ khi sinh ra cho đến khi chết. Quyền, nghĩa vụ dân sự của cá nhân có thể phát sinh trước khi sinh ra hoặc còn tồn tại sau khi chết trong trường hợp luật có quy định.",
            article_number="Điều 18", 
            law_name="Bộ luật Dân sự 2015"
        )
    ]

    print("=== HỆ THỐNG MULTI-STAGE RETRIEVAL CHO ALQAC 2025 ===\n")

    # Khởi tạo hệ thống (không cần API key cho demo)
    system = MultiStageRetrievalSystem(sample_documents)

    # Câu hỏi mẫu
    test_questions = [
        "Khi nào cá nhân có quyền, nghĩa vụ dân sự?",
        "Bộ luật Dân sự quy định về những vấn đề gì?",
        "Quyền dân sự có thể tồn tại sau khi chết không?"
    ]

    for question in test_questions:
        print(f"Câu hỏi: {question}")
        print("-" * 50)

        result = system.search_and_answer(question)

        print(f"Câu trả lời:\n{result['answer']}\n")
        print(f"Số văn bản tìm thấy ở giai đoạn 1: {result['stages']['stage1_count']}")
        print(f"Số văn bản sau re-ranking: {result['stages']['stage2_count']}")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
