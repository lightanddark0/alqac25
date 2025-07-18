import json
import random

INPUT_FILE = 'alqac25_train.json'
TRAIN_FILE = 'alqac25_train_split.json'
TEST_FILE = 'alqac25_test_split.json'
SPLIT_RATIO = 0.8  # 80% train, 20% test

# Đọc dữ liệu
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Xáo trộn dữ liệu
random.shuffle(data)

# Chia train/test
split_idx = int(len(data) * SPLIT_RATIO)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Ghi file train
with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# Ghi file test
with open(TEST_FILE, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Đã chia dữ liệu: {len(train_data)} train, {len(test_data)} test.") 