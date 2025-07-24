import json

INPUT_FILE = 'alqac25_Task_1.json'
OUTPUT_FILE = 'alqac25_Task_1_top1.json'

with open(INPUT_FILE, encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if 'relevant_articles' in item and isinstance(item['relevant_articles'], list):
        item['relevant_articles'] = item['relevant_articles'][:1]

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Đã lưu file chỉ lấy top 2 tài liệu: {OUTPUT_FILE}") 