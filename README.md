# Vietnamese Legal QA System

Hệ thống Hỏi đáp Pháp luật Tiếng Việt sử dụng kỹ thuật embedding và retrieval để tìm kiếm và trả lời các câu hỏi liên quan đến pháp luật Việt Nam.

## 🚀 Tính năng

- **Hybrid Retrieval**: Kết hợp BM25 và Sentence Transformers để tìm kiếm tài liệu pháp luật liên quan
- **Vietnamese Embedding**: Sử dụng mô hình `truro7/vn-law-embedding` được tối ưu cho tiếng Việt
- **GPU Optimization**: Hỗ trợ chạy trên GPU để tăng tốc độ xử lý
- **Multi-stage Processing**: Xử lý đa giai đoạn với BM25 pre-ranking và dense reranking

## 📁 Cấu trúc dự án

```
├── vietnamese_embedding.py      # Main script cho embedding và retrieval
├── main.py                      # Multi-stage retrieval system
├── requirements.txt             # Dependencies cho main.py
├── requirements_vietnamese_embedding.txt  # Dependencies cho vietnamese_embedding.py
├── alqac25_law.json            # Corpus pháp luật
├── alqac25_private_test_Task_1.json  # Dataset test
├── check_gpu.py                # Script kiểm tra GPU
└── README.md                   # File này
```

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA 12.1+ (nếu sử dụng GPU)
- NVIDIA GPU (khuyến nghị)

### Cài đặt dependencies

**Cho vietnamese_embedding.py:**
```bash
pip install -r requirements_vietnamese_embedding.txt
```

**Cho main.py:**
```bash
pip install -r requirements.txt
```

### Cài đặt PyTorch với CUDA support

**Sử dụng pip:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Hoặc sử dụng conda:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 🚀 Sử dụng

### 1. Kiểm tra GPU
```bash
python check_gpu.py
```

### 2. Chạy Vietnamese Embedding
```bash
python vietnamese_embedding.py
```

Script này sẽ:
- Tải corpus pháp luật từ `alqac25_law.json`
- Tải câu hỏi test từ `alqac25_private_test_Task_1.json`
- Tạo BM25 index và FAISS dense index
- Tìm các tài liệu liên quan cho từng câu hỏi
- Lưu kết quả vào `questions_with_relevant_articles.json`

### 3. Chạy Multi-stage Retrieval
```bash
python main.py
```

## 📊 Kết quả

- `questions_with_relevant_articles.json`: Danh sách các tài liệu pháp luật liên quan cho từng câu hỏi
- `law_embeds.npy`: Embeddings của corpus pháp luật
- `faiss.index`: FAISS index cho vector search

## 🔧 Cấu hình

### GPU Configuration
File `vietnamese_embedding.py` tự động phát hiện GPU và tối ưu hóa:
- Tự động chuyển sang CPU nếu không có GPU
- Tối ưu batch size cho GPU
- Quản lý memory GPU

### Model Configuration
- **Embedding Model**: `truro7/vn-law-embedding`
- **BM25 Parameters**: k1=1.2, b=0.75
- **Top-k Retrieval**: 3 documents per query

## 📈 Performance

- **GPU Acceleration**: Tăng tốc độ xử lý 3-5x so với CPU
- **Hybrid Retrieval**: Cải thiện độ chính xác retrieval
- **Vietnamese Optimization**: Tối ưu cho văn bản tiếng Việt

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Dự án này được phân phối dưới MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue trên GitHub.

---

**Lưu ý**: Đảm bảo bạn có đủ quyền truy cập vào các file dữ liệu pháp luật trước khi sử dụng hệ thống này. 