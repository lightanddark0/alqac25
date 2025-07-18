import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ========== CONFIG ==========
LAW_PATH = "alqac25_law.json"
QA_PATH = "alqac25_private_test_Task_1.json"  # Đánh giá trên bộ test
TOP_K = 1  # Số documents top-k sau khi rerank
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Tăng batch size cho GPU
# ============================

def load_laws(law_path):
    with open(law_path, encoding="utf-8") as f:
        laws = json.load(f)
    docs = []
    for law in laws:
        law_name = law["id"]
        for article in law["articles"]:
            docs.append({
                "law_id": law_name,
                "article_id": article["id"],
                "text": article["text"]
            })
    return docs

def load_questions(qa_path):
    with open(qa_path, encoding="utf-8") as f:
        return json.load(f)

def build_bm25(docs):
    corpus = [doc["text"] for doc in docs]
    tokenized_corpus = [text.split() for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def build_dense_index(docs, embedder):
    corpus = [doc["text"] for doc in docs]
    print(f"Encoding {len(corpus)} documents with batch_size={BATCH_SIZE} on {DEVICE}...")
    embeddings = embedder.encode(
        corpus, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    dim = embeddings.shape[1]
    
    # Sử dụng GPU index nếu có GPU
    if DEVICE == "cuda":
        # Tạo index trên CPU trước, sau đó chuyển lên GPU nếu cần
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f"Created FAISS index with {len(embeddings)} vectors of dimension {dim}")
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f"Created FAISS index with {len(embeddings)} vectors of dimension {dim}")
    
    # Lưu
    np.save("law_embeds.npy", embeddings)
    faiss.write_index(index, "faiss.index")
    return index, embeddings

def hybrid_retrieve(question, bm25, docs, embedder, dense_index, dense_corpus_embeds, top_k=3, alpha=0.5):
    # BM25: lấy 10 tài liệu
    bm25_scores = bm25.get_scores(question.split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:5]  # Lấy top 10 từ BM25
    
    # Rerank bằng SentenceTransformer
    bm25_docs = [docs[idx] for idx in bm25_top_idx]
    bm25_texts = [doc["text"] for doc in bm25_docs]
    
    # Encode question và documents với GPU optimization
    question_emb = embedder.encode([question], convert_to_numpy=True, device=DEVICE)
    doc_embs = embedder.encode(bm25_texts, convert_to_numpy=True, device=DEVICE)
    
    # Tính cosine similarity
    similarities = np.dot(doc_embs, question_emb.T).flatten()
    
    # Sắp xếp theo similarity và lấy top 2
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Trả về top 2 documents
    results = [bm25_docs[idx] for idx in top_indices]
    return results

def build_prompt(article_content, question, prompt_type="Đúng/Sai"):
    if prompt_type == "Đúng/Sai":
        prompt = (
            "Dựa nội dung sau đây, hãy xác định câu sau là Đúng hay Sai.\n\n"
            "Lưu ý: bắt buộc chỉ trả lời Đúng hay Sai không cần giải thích gì thêm.\n\n"
            "Ví dụ: câu hỏi là Hội đồng trọng tài có thể không dùng pháp luật Việt Nam để giải quyết tranh chấp, đúng hay sai?\n\n"
            "Thì đáp án: Đúng\n\n"
            f"Câu hỏi: {question}\n\n{article_content}\n\n"
            "Đáp án: "
        )
    elif prompt_type == "Trắc nghiệm":
        prompt = (
            "Dựa vào nội dung sau đây, hãy chọn đáp án đúng.\n\n"
            "Lưu ý: bắt buộc chỉ cần trả lời bằng một trong các lựa chọn A, B, C, hoặc D không cần giải thích gì thêm.\n\n"
            "Ví dụ: câu hỏi là Người xem dưới 16 tuổi được xem phim có nội dung thuộc phân loại nào sau đây? A. T18 (18+), B. T16 (16+), C. C, D. K\n\n"
            "Thì đáp án: D\n\n"
            f"Câu hỏi: {question}\n\n{article_content}\n\n"
            "Đáp án: "
        )
    elif prompt_type == "Tự luận":
        prompt = (
            "Dựa nội dung sau đây, hãy trả lời ngắn gọn, không giải thích gì thêm\n\n"
            "Ví dụ: câu hỏi là Hành vi nào liên quan đến phiên dịch được coi là hành vi cản trở thu thập xác minh chứng cớ của tòa án?\n\n"
            "Thì đáp án là Dịch sai sự thật\n\n"
            f"Câu hỏi: {question}\n\n{article_content}\n\n"
            "Đáp án: "
        )
    else:
        raise ValueError("prompt_type không hợp lệ")
    return prompt

def find_relevant_articles_for_questions(questions, docs, bm25, embedder, dense_index, dense_corpus_embeds, top_k=3):
    results = []
    print(f"Processing {len(questions)} questions on {DEVICE}...")
    for qa in tqdm(questions, desc="Tìm tài liệu liên quan"):
        question = qa["text"]
        top_docs = hybrid_retrieve(question, bm25, docs, embedder, dense_index, dense_corpus_embeds, top_k=top_k)
        relevant_articles = [
            {"law_id": doc["law_id"], "article_id": doc["article_id"]} for doc in top_docs
        ]
        results.append({
            "question_id": qa.get("question_id"),
            "text": question,
            "relevant_articles": relevant_articles
        })
    return results

def main():
    print(f"Running on device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("Tải dữ liệu...")
    docs = load_laws(LAW_PATH)
    questions = load_questions(QA_PATH)  # Đánh giá toàn bộ bộ test

    print("Khởi tạo BM25...")
    bm25 = build_bm25(docs)

    print(f"Tải mô hình embedding truro7/vn-law-embedding trên {DEVICE}...")
    embedder = SentenceTransformer("truro7/vn-law-embedding", device=DEVICE)

    print("Xây dựng FAISS dense index...")
    dense_index, dense_corpus_embeds = build_dense_index(docs, embedder)

    # Chỉ thực hiện tìm và lưu relevant_articles cho từng câu hỏi
    questions_with_rels = find_relevant_articles_for_questions(questions, docs, bm25, embedder, dense_index, dense_corpus_embeds, top_k=3)
    with open("questions_with_relevant_articles.json", "w", encoding="utf-8") as f:
        json.dump(questions_with_rels, f, ensure_ascii=False, indent=2)
    print("\nĐã lưu danh sách relevant_articles cho từng câu hỏi vào questions_with_relevant_articles.json")
    
    # Clear GPU memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print("Đã xóa GPU memory cache")

if __name__ == "__main__":
    main()

