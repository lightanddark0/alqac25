# BGE Model Fine-tuning

Fine-tuning BGE (BAAI General Embedding) model cho tác vụ retrieval pháp luật tiếng Việt.

## 🚀 Quick Start

### Cách 1: Setup tự động (Khuyến nghị)
```bash
python setup_finetune_bge.py
```

### Cách 2: Setup thủ công
```bash
# 1. Cài đặt PyTorch với CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Cài đặt các dependencies
pip install -r requirements_finetune_bge.txt
```

## 📋 Requirements

### Hệ thống
- Python 3.8+
- CUDA 12.1+ (khuyến nghị)
- NVIDIA GPU với ít nhất 4GB VRAM
- RAM: 8GB+

### Dependencies
Xem file `requirements_finetune_bge.txt` để biết chi tiết.

## 🔧 Configuration

File `finetune_bge.py` có các cấu hình sau:

```python
# Model configuration
MODEL_NAME = 'BAAI/bge-small-en-v1.5'  # Model gốc
BATCH_SIZE = 8  # Batch size cho GPU
EPOCHS = 2      # Số epoch training
LR = 2e-5       # Learning rate

# Data files
CORPUS_FILES = ['alqac25_law.json']           # Corpus pháp luật
TRAIN_FILES = ['alqac25_train.json']          # Training data
TEST_FILE = 'alqac25_train_split.json'        # Test data
```

## 📊 Data Format

### Corpus Format
```json
[
  {
    "id": "law_id",
    "articles": [
      {
        "id": "article_id",
        "text": "Nội dung điều luật..."
      }
    ]
  }
]
```

### Training Data Format
```json
[
  {
    "id": "question_id",
    "text": "Câu hỏi pháp luật",
    "relevant_articles": [
      {
        "law_id": "law_id",
        "article_id": "article_id"
      }
    ]
  }
]
```

## 🏃‍♂️ Usage

### Chạy fine-tuning
```bash
python finetune_bge.py
```

### Output
- **Model**: `bge_finetuned_model/` - Model đã fine-tune
- **Results**: `bge_finetune_results.json` - Kết quả đánh giá

## 📈 Performance

### Metrics
- **Precision@2**: Độ chính xác trong top 2 kết quả
- **Recall@2**: Độ bao phủ trong top 2 kết quả  
- **F2 Score**: F-score với beta=2 (ưu tiên recall)

### Expected Results
- Precision: ~0.6-0.8
- Recall: ~0.4-0.6
- F2 Score: ~0.5-0.7

## 🔍 Troubleshooting

### Lỗi thường gặp

1. **CUDA out of memory**
   ```bash
   # Giảm batch size trong finetune_bge.py
   BATCH_SIZE = 4  # Thay vì 8
   ```

2. **Model loading error**
   ```bash
   # Kiểm tra kết nối internet
   # Hoặc tải model offline
   ```

3. **Data format error**
   ```bash
   # Kiểm tra format JSON
   # Đảm bảo file tồn tại
   ```

### GPU Optimization
- Sử dụng `use_amp=True` cho mixed precision
- `pin_memory=True` cho DataLoader
- Batch size tối ưu cho GPU memory

## 📝 Notes

- Model sẽ được lưu trong thư mục `bge_finetuned_model/`
- Kết quả đánh giá được lưu trong `bge_finetune_results.json`
- GPU memory sẽ được tự động clear sau khi hoàn thành
- Có thể điều chỉnh hyperparameters trong file config

## 🤝 Contributing

Nếu gặp vấn đề, hãy:
1. Kiểm tra GPU và CUDA setup
2. Xem log lỗi chi tiết
3. Thử giảm batch size
4. Kiểm tra format dữ liệu 