import json
with open('alqac25_private_test_task2.json', encoding='utf-8') as f:
    data = json.load(f)
for item in data:
    if 'answer' not in item or item['answer'] == '' or item['answer'] is None:
        qtype = item.get('question_type')
        if qtype == 'Đúng/Sai':
            item['answer'] = 'Đúng'
        elif qtype == 'Trắc nghiệm':
            item['answer'] = 'A'
        elif qtype == 'Tự luận':
            item['answer'] = 'Chưa đủ thông tin'
with open('alqac25_private_test_task2.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
