import re
import matplotlib.pyplot as plt
import os

def parse_logs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    models_data = {
        'Baseline': {'loss': [], 'recall': [], 'ndcg': []},
        'Biclique GCN': {'loss': [], 'recall': [], 'ndcg': []},
        'Biclique CL': {'loss': [], 'recall': [], 'ndcg': []},
        'Full TSB-CL': {'loss': [], 'recall': [], 'ndcg': []}
    }

    # Define regex patterns
    # Pattern to identify the start of a model section
    model_start_patterns = {
        'Baseline': r'Running Pure LightGCN \(Baseline\)',
        'Biclique GCN': r'Running Biclique GCN \(No CL, No RNN\)',
        'Biclique CL': r'Running Biclique \+ CL \(No RNN\)',
        'Full TSB-CL': r'Running Full TSB-CL \(Biclique \+ CL \+ RNN\)'
    }

    # Pattern to extract metrics
    # Epoch 1/50: Loss = 0.5714 (6.38s)
    # Test Recall@20: 0.0508, NDCG@20: 0.0362
    epoch_pattern = re.compile(r'Epoch (\d+)/50: Loss = ([\d\.]+)')
    metrics_pattern = re.compile(r'Test Recall@20: ([\d\.]+), NDCG@20: ([\d\.]+)')

    current_model = None
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Check for model start
        for model_name, pattern in model_start_patterns.items():
            if re.search(pattern, line):
                current_model = model_name
                print(f"Found start of {model_name} at line {i+1}")
                break
        
        if current_model:
            # Extract Loss
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                loss = float(epoch_match.group(2))
                models_data[current_model]['loss'].append(loss)
                
                # Look ahead for metrics in the next few lines
                # Usually it's in the next line or the one after (due to snapshot logs)
                for j in range(1, 10): # Look ahead up to 10 lines
                    if i + j < len(lines):
                        next_line = lines[i+j]
                        metrics_match = metrics_pattern.search(next_line)
                        if metrics_match:
                            recall = float(metrics_match.group(1))
                            ndcg = float(metrics_match.group(2))
                            models_data[current_model]['recall'].append(recall)
                            models_data[current_model]['ndcg'].append(ndcg)
                            break

    return models_data

def plot_metrics(models_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = range(1, 51)
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    for model_name, data in models_data.items():
        if len(data['loss']) > 0:
            # Ensure we match the length of epochs if training was interrupted or parsed incorrectly
            x_axis = range(1, len(data['loss']) + 1)
            plt.plot(x_axis, data['loss'], label=model_name)
    
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
    plt.close()

    # Plot Recall
    plt.figure(figsize=(10, 6))
    for model_name, data in models_data.items():
        if len(data['recall']) > 0:
            x_axis = range(1, len(data['recall']) + 1)
            plt.plot(x_axis, data['recall'], label=model_name)

    plt.title('Test Recall@20 over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'recall_comparison.png'))
    plt.close()

    # Plot NDCG
    plt.figure(figsize=(10, 6))
    for model_name, data in models_data.items():
        if len(data['ndcg']) > 0:
            x_axis = range(1, len(data['ndcg']) + 1)
            plt.plot(x_axis, data['ndcg'], label=model_name)

    plt.title('Test NDCG@20 over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('NDCG@20')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ndcg_comparison.png'))
    plt.close()

if __name__ == "__main__":
    log_file = r"c:\Users\LENOVO\Desktop\论文代码\ai_project\结果文件.txt"
    output_dir = r"c:\Users\LENOVO\Desktop\论文代码\ai_project\plots"
    
    print("Parsing logs...")
    data = parse_logs(log_file)
    
    # Print summary of parsed data
    for model, metrics in data.items():
        print(f"{model}: {len(metrics['loss'])} epochs parsed.")
        if len(metrics['loss']) > 0:
            print(f"  Final Loss: {metrics['loss'][-1]}")
            print(f"  Final Recall: {metrics['recall'][-1]}")
            print(f"  Final NDCG: {metrics['ndcg'][-1]}")

    print("Generating plots...")
    plot_metrics(data, output_dir)
    print(f"Plots saved to {output_dir}")
