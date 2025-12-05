import re

def parse_best_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    models_data = {
        'Baseline': {'best_recall': 0, 'best_ndcg': 0, 'best_epoch': 0},
        'Biclique GCN': {'best_recall': 0, 'best_ndcg': 0, 'best_epoch': 0},
        'Biclique CL': {'best_recall': 0, 'best_ndcg': 0, 'best_epoch': 0},
        'Full TSB-CL': {'best_recall': 0, 'best_ndcg': 0, 'best_epoch': 0}
    }

    model_start_patterns = {
        'Baseline': r'Running Pure LightGCN \(Baseline\)',
        'Biclique GCN': r'Running Biclique GCN \(No CL, No RNN\)',
        'Biclique CL': r'Running Biclique \+ CL \(No RNN\)',
        'Full TSB-CL': r'Running Full TSB-CL \(Biclique \+ CL \+ RNN\)'
    }

    epoch_pattern = re.compile(r'Epoch (\d+)/50')
    metrics_pattern = re.compile(r'Test Recall@20: ([\d\.]+), NDCG@20: ([\d\.]+)')
    final_result_pattern = re.compile(r'Result for (\w+): Recall@20 = ([\d\.]+), NDCG@20 = ([\d\.]+)')

    current_model = None
    current_epoch = 0
    
    lines = content.split('\n')
    
    # Check for explicit final results at the end of the file
    print("--- Explicit Final Test Results in Log ---")
    found_final = False
    for line in lines:
        match = final_result_pattern.search(line)
        if match:
            found_final = True
            print(f"Model: {match.group(1)}, Recall: {match.group(2)}, NDCG: {match.group(3)}")
    if not found_final:
        print("No explicit final test results found.")
    print("------------------------------------------\n")

    # Parse training logs for best epoch
    for i, line in enumerate(lines):
        # Check for model start
        for model_name, pattern in model_start_patterns.items():
            if re.search(pattern, line):
                current_model = model_name
                break
        
        if current_model:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            metrics_match = metrics_pattern.search(line)
            if metrics_match:
                recall = float(metrics_match.group(1))
                ndcg = float(metrics_match.group(2))
                
                if recall > models_data[current_model]['best_recall']:
                    models_data[current_model]['best_recall'] = recall
                    models_data[current_model]['best_ndcg'] = ndcg
                    models_data[current_model]['best_epoch'] = current_epoch

    return models_data

if __name__ == "__main__":
    log_file = r"c:\Users\LENOVO\Desktop\论文代码\ai_project\结果文件.txt"
    best_results = parse_best_results(log_file)
    
    print("--- Best Test Results found in Training Logs (Peak Performance) ---")
    for model, metrics in best_results.items():
        print(f"{model}: Best Recall@20 = {metrics['best_recall']} (Epoch {metrics['best_epoch']}), NDCG@20 = {metrics['best_ndcg']}")
