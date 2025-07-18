import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity

# ----------- CONFIG -----------
CORPUS_FILES = [
    'alqac25_law.json'  # Chỉ sử dụng file này
]
TRAIN_FILES = [
    'alqac25_train.json'  # Chỉ sử dụng file này
]
TEST_FILE = 'alqac25_train_split.json'
MODEL_NAME = 'BAAI/bge-small-en-v1.5'  # Thay bằng model BGE tiếng Việt nếu có
BATCH_SIZE = 8  # Tăng batch size cho GPU
EPOCHS = 2
LR = 2e-5
OUTPUT_DIR = 'bge_finetuned_model'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------- DATA PREP -----------
def load_corpus(files):
    """
    Load corpus từ các file JSON với định dạng:
    [{"id": "law_id", "articles": [{"id": "article_id", "text": "content"}]}]
    """
    corpus = {}
    for file in files:
        if os.path.exists(file):  # Kiểm tra file tồn tại
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for law in data:
                    law_id = law['id']
                    for article in law['articles']:
                        article_id = article['id']
                        # Tạo key duy nhất: law_id + article_id
                        doc_id = f"{law_id}_{article_id}"
                        corpus[doc_id] = article['text']
        else:
            print(f"Warning: File {file} not found, skipping...")
    return corpus

def load_train_pairs(train_files, corpus):
    """
    Load training pairs từ các file JSON với định dạng:
    [{"id": "question_id", "text": "question", "relevant_articles": [{"law_id": "law_id", "article_id": "article_id"}]}]
    """
    examples = []
    for train_file in train_files:
        if os.path.exists(train_file):  # Kiểm tra file tồn tại
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                query = item.get('text', item.get('question', ''))
                relevant_articles = item.get('relevant_articles', [])
                
                for article in relevant_articles:
                    law_id = article.get('law_id')
                    article_id = article.get('article_id')
                    if law_id and article_id:
                        # Tạo doc_id tương ứng với corpus
                        doc_id = f"{law_id}_{article_id}"
                        if doc_id in corpus:
                            examples.append(InputExample(texts=[query, corpus[doc_id]]))
        else:
            print(f"Warning: File {train_file} not found, skipping...")
    return examples

def load_test_set(test_file, corpus):
    """
    Load test set từ file JSON với định dạng tương tự train
    """
    if not os.path.exists(test_file):
        print(f"Warning: File {test_file} not found, returning empty test set...")
        return []
        
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_queries = []
    for item in data:
        query = item.get('text', item.get('question', ''))
        relevant_articles = item.get('relevant_articles', [])
        doc_ids = []
        
        for article in relevant_articles:
            law_id = article.get('law_id')
            article_id = article.get('article_id')
            if law_id and article_id:
                # Tạo doc_id tương ứng với corpus
                doc_id = f"{law_id}_{article_id}"
                if doc_id in corpus:
                    doc_ids.append(doc_id)
        
        test_queries.append((query, doc_ids))
    return test_queries

def encode_in_chunks(model, texts, batch_size=16, device=None):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding chunks"):
        batch = texts[i:i+batch_size]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device
        )
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)

# ----------- MAIN PIPELINE -----------
def main():
    print(f'Using device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    print('Loading corpus...')
    corpus = load_corpus(CORPUS_FILES)
    print(f'Corpus size: {len(corpus)}')

    print('Loading train pairs...')
    train_examples = load_train_pairs(TRAIN_FILES, corpus)
    print(f'Train pairs: {len(train_examples)}')

    if len(train_examples) == 0:
        print("No training examples found! Please check your data files.")
        return

    print('Loading test set...')
    test_queries = load_test_set(TEST_FILE, corpus)
    print(f'Test queries: {len(test_queries)}')

    # Load model
    print('Loading model...')
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # DataLoader với tối ưu hóa GPU
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=BATCH_SIZE,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Training
    print('Start training...')
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_DIR,
        use_amp=True if DEVICE == 'cuda' else False,  # Tăng tốc và giảm RAM khi dùng GPU
        optimizer_params={'lr': LR}
    )
    print(f'Model saved to {OUTPUT_DIR}')

    # Evaluation
    if len(test_queries) > 0:
        print('Evaluating...')
        model = SentenceTransformer(OUTPUT_DIR, device=DEVICE)
        queries = [q for q, _ in test_queries]
        gt_doc_ids = [doc_ids for _, doc_ids in test_queries]
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]

        # Encode corpus theo chunk trên GPU
        print('Encoding corpus in chunks...')
        corpus_emb = encode_in_chunks(model, corpus_texts, batch_size=BATCH_SIZE, device=DEVICE)
        # Encode queries theo chunk trên GPU
        print('Encoding queries in chunks...')
        query_emb = encode_in_chunks(model, queries, batch_size=BATCH_SIZE//2, device=DEVICE)

        # Compute similarity and evaluate precision, recall, F2 score
        from sklearn.metrics import precision_score, recall_score, f1_score
        import numpy as np
        
        all_precisions = []
        all_recalls = []
        all_f2_scores = []
        
        for i, emb in enumerate(tqdm(query_emb, desc="Evaluating")):
            scores = cosine_similarity(emb.cpu().numpy().reshape(1, -1), corpus_emb.cpu().numpy())[0]
            top_k = scores.argsort()[-2:][::-1]  # Top 2 results
            top_doc_ids = [corpus_ids[idx] for idx in top_k]
            
            # Ground truth labels
            gt_docs = set(gt_doc_ids[i])
            
            # Create binary labels for top 2 results
            y_true = []
            y_pred = []
            for doc_id in top_doc_ids:
                y_pred.append(1)  # Predicted as relevant
                y_true.append(1 if doc_id in gt_docs else 0)  # Actually relevant
            
            # Calculate metrics for this query
            if len(y_true) > 0:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                if precision <= 0 and recall <= 0:
                    f2 = 0
                else:
                    f2 = 5 * (precision * recall) / (4*precision + recall)  # F2 score (beta=2)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f2_scores.append(f2)
        
        # Calculate average metrics
        if len(all_precisions) > 0:
            avg_precision = np.mean(all_precisions)
            avg_recall = np.mean(all_recalls)
            avg_f2 = np.mean(all_f2_scores)
            
            print(f"\n=== EVALUATION RESULTS ===")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            print(f"Average F2 Score: {avg_f2:.4f}")
            
            # Lưu kết quả
            results = {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f2': avg_f2,
                'num_test_queries': len(test_queries),
                'num_train_pairs': len(train_examples)
            }
            
            with open('bge_finetune_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print("Results saved to bge_finetune_results.json")
        else:
            print("No valid evaluation results!")
    else:
        print("No test queries available for evaluation!")
    
    # Clear GPU memory
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        print("GPU memory cleared")

if __name__ == "__main__":
    main() 