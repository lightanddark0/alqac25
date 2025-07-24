import json

FILE1 = 'alqac25_private_test_Task_1.json'
FILE2 = 'alqac25_Task_1.json'
OUTPUT_FILE = 'alqac25_private_test_Task_1_intersection.json'

with open(FILE1, encoding='utf-8') as f:
    data1 = json.load(f)
with open(FILE2, encoding='utf-8') as f:
    data2 = json.load(f)

# Tạo dict để tra cứu theo question_id
q2arts2 = {item['question_id']: item['relevant_articles'] for item in data2}

for item in data1:
    qid = item['question_id']
    arts1 = item['relevant_articles']
    arts2 = q2arts2.get(qid, [])
    set1 = set((a['law_id'], a['article_id']) for a in arts1)
    set2 = set((a['law_id'], a['article_id']) for a in arts2)
    inter = set1 & set2
    if inter:
        item['relevant_articles'] = [
            {'law_id': law_id, 'article_id': article_id}
            for (law_id, article_id) in inter
        ]
    else:
        # Nếu giao rỗng, giữ lại toàn bộ relevant_articles từ file1
        item['relevant_articles'] = arts1

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data1, f, ensure_ascii=False, indent=2)

print(f"Đã lưu file giao tài liệu (nếu rỗng thì lấy file1): {OUTPUT_FILE}") 