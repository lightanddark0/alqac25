# run_bge_m3_experiment.py - Script chạy thử nghiệm BGE-M3

import json
import time
import logging
from pathlib import Path
from bge_m3_alqac import BGE_M3ALQACSystem
from main import MultiStageRetrievalSystem

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_bge_m3_experiment():
    """Chạy thử nghiệm với BGE-M3"""
    print("="*60)
    print("THỬ NGHIỆM BGE-M3 CHO ALQAC 2025")
    print("="*60)
    
    # Cấu hình BGE-M3
    bge_config = {
        'bge_model': 'BAAI/bge-m3',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'top_k': 3,  # Tối ưu cho precision
        'llm_local_model_dir': './finetune_standard'
    }
    
    # Khởi tạo hệ thống BGE-M3
    print("\n1. Khởi tạo hệ thống BGE-M3...")
    bge_system = BGE_M3ALQACSystem(bge_config)
    
    # Tải dữ liệu
    print("\n2. Tải dữ liệu...")
    bge_system.load_data('alqac25_test_split.json')
    
    # Xây dựng index
    print("\n3. Xây dựng BGE-M3 index...")
    start_time = time.time()
    bge_system.build_index()
    index_time = time.time() - start_time
    print(f"   Thời gian xây dựng index: {index_time:.2f}s")
    
    # Xử lý câu hỏi
    print("\n4. Xử lý câu hỏi với BGE-M3...")
    start_time = time.time()
    bge_results = bge_system.process_all_questions()
    processing_time = time.time() - start_time
    print(f"   Thời gian xử lý: {processing_time:.2f}s")
    
    # Lưu kết quả
    print("\n5. Lưu kết quả BGE-M3...")
    bge_system.save_results(bge_results, 'bge_m3_results.json')
    
    return bge_results, bge_system.evaluator.get_summary_metrics()

def run_original_experiment():
    """Chạy thử nghiệm với hệ thống gốc"""
    print("\n" + "="*60)
    print("THỬ NGHIỆM HỆ THỐNG GỐC (BM25 + BERT)")
    print("="*60)
    
    # Cấu hình hệ thống gốc
    original_config = {
        'bm25_k1': 2,
        'bm25_b': 0.75,
        'bm25_top_k': 10,
        'bert_model': 'vinai/phobert-base',
        'bert_top_k': 3,
        'bert_alpha': 0.3,
        'device': 'cpu',
        'llm_api_key': None,
        'llm_model': 'llama3.2-7b-instruct',
        'llm_local_model_dir': './finetune_standard'
    }
    
    # Khởi tạo hệ thống gốc
    print("\n1. Khởi tạo hệ thống gốc...")
    original_system = MultiStageRetrievalSystem(original_config)
    
    # Tải dữ liệu
    print("\n2. Tải dữ liệu...")
    original_system.load_data('alqac25_test_split.json')
    
    # Xây dựng index
    print("\n3. Xây dựng index (BM25)...")
    start_time = time.time()
    original_system.build_index()
    index_time = time.time() - start_time
    print(f"   Thời gian xây dựng index: {index_time:.2f}s")
    
    # Xử lý câu hỏi
    print("\n4. Xử lý câu hỏi với hệ thống gốc...")
    start_time = time.time()
    original_results = original_system.process_all_questions()
    processing_time = time.time() - start_time
    print(f"   Thời gian xử lý: {processing_time:.2f}s")
    
    # Lưu kết quả
    print("\n5. Lưu kết quả hệ thống gốc...")
    original_system.save_results(original_results, 'original_results.json')
    
    return original_results, original_system.evaluator.get_summary_metrics()

def compare_results(bge_metrics, original_metrics):
    """So sánh kết quả giữa hai hệ thống"""
    print("\n" + "="*60)
    print("SO SÁNH KẾT QUẢ")
    print("="*60)
    
    print("\nTask 1 - Legal Document Retrieval:")
    print("-" * 40)
    
    if 'task1' in bge_metrics and 'task1' in original_metrics:
        bge_precision = bge_metrics['task1']['avg_precision']
        original_precision = original_metrics['task1']['avg_precision']
        precision_improvement = ((bge_precision - original_precision) / original_precision) * 100
        
        bge_recall = bge_metrics['task1']['avg_recall']
        original_recall = original_metrics['task1']['avg_recall']
        recall_improvement = ((bge_recall - original_recall) / original_recall) * 100
        
        bge_f2 = bge_metrics['task1']['avg_f2']
        original_f2 = original_metrics['task1']['avg_f2']
        f2_improvement = ((bge_f2 - original_f2) / original_f2) * 100
        
        print(f"Precision:")
        print(f"  BGE-M3:     {bge_precision:.4f}")
        print(f"  Original:   {original_precision:.4f}")
        print(f"  Cải thiện:  {precision_improvement:+.2f}%")
        
        print(f"\nRecall:")
        print(f"  BGE-M3:     {bge_recall:.4f}")
        print(f"  Original:   {original_recall:.4f}")
        print(f"  Cải thiện:  {recall_improvement:+.2f}%")
        
        print(f"\nF2-score:")
        print(f"  BGE-M3:     {bge_f2:.4f}")
        print(f"  Original:   {original_f2:.4f}")
        print(f"  Cải thiện:  {f2_improvement:+.2f}%")
    
    print("\nTask 2 - Legal Question Answering:")
    print("-" * 40)
    
    if 'task2' in bge_metrics and 'task2' in original_metrics:
        bge_accuracy = bge_metrics['task2']['accuracy']
        original_accuracy = original_metrics['task2']['accuracy']
        accuracy_improvement = ((bge_accuracy - original_accuracy) / original_accuracy) * 100
        
        print(f"Accuracy:")
        print(f"  BGE-M3:     {bge_accuracy:.4f}")
        print(f"  Original:   {original_accuracy:.4f}")
        print(f"  Cải thiện:  {accuracy_improvement:+.2f}%")
        
        print(f"\nCorrect Answers:")
        print(f"  BGE-M3:     {bge_metrics['task2']['correct_answers']}/{bge_metrics['task2']['total_questions']}")
        print(f"  Original:   {original_metrics['task2']['correct_answers']}/{original_metrics['task2']['total_questions']}")

def save_comparison_report(bge_metrics, original_metrics):
    """Lưu báo cáo so sánh"""
    comparison_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'bge_m3_metrics': bge_metrics,
        'original_metrics': original_metrics,
        'summary': {
            'bge_m3_better_precision': False,
            'bge_m3_better_recall': False,
            'bge_m3_better_f2': False,
            'bge_m3_better_accuracy': False
        }
    }
    
    # So sánh và cập nhật summary
    if 'task1' in bge_metrics and 'task1' in original_metrics:
        comparison_data['summary']['bge_m3_better_precision'] = (
            bge_metrics['task1']['avg_precision'] > original_metrics['task1']['avg_precision']
        )
        comparison_data['summary']['bge_m3_better_recall'] = (
            bge_metrics['task1']['avg_recall'] > original_metrics['task1']['avg_recall']
        )
        comparison_data['summary']['bge_m3_better_f2'] = (
            bge_metrics['task1']['avg_f2'] > original_metrics['task1']['avg_f2']
        )
    
    if 'task2' in bge_metrics and 'task2' in original_metrics:
        comparison_data['summary']['bge_m3_better_accuracy'] = (
            bge_metrics['task2']['accuracy'] > original_metrics['task2']['accuracy']
        )
    
    with open('comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nBáo cáo so sánh đã được lưu vào 'comparison_report.json'")

def main():
    """Hàm main để chạy thử nghiệm"""
    try:
        # Chạy thử nghiệm BGE-M3
        bge_results, bge_metrics = run_bge_m3_experiment()
        
        # Chạy thử nghiệm hệ thống gốc
        original_results, original_metrics = run_original_experiment()
        
        # So sánh kết quả
        compare_results(bge_metrics, original_metrics)
        
        # Lưu báo cáo so sánh
        save_comparison_report(bge_metrics, original_metrics)
        
        print("\n" + "="*60)
        print("HOÀN THÀNH THỬ NGHIỆM")
        print("="*60)
        print("Các file kết quả:")
        print("- bge_m3_results.json: Kết quả BGE-M3")
        print("- original_results.json: Kết quả hệ thống gốc")
        print("- comparison_report.json: Báo cáo so sánh")
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy thử nghiệm: {e}")
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    import torch
    main() 