import torch
import numpy as np
from imports.TrainModel import TrainModel
from tuner import model2
from tuner import model2_5l
from tuner import model2_3l
from tuner import model2_2l
from torch_geometric.transforms import RandomJitter, Compose, Cartesian
import os
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score
import random

#cuda error workaround
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

#set seed=0 for reproducibility
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

#save confusion matrix construction data
def save_cm_data(predictions, true_labels, graph_type, test_loss, log_dir, timestamp):
    """Save confusion matrix construction data to txt file"""
    if predictions is None or true_labels is None:
        print(f"Warning: No predictions/labels for {graph_type}")
        return
    
    cm_file = log_dir / f"cm_{graph_type}_{timestamp}.txt"
    with open(cm_file, 'w') as f:
        f.write(f"Graph Type: {graph_type}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n") 
        f.write(f"Total Samples: {len(predictions)}\n\n")
        f.write("Predictions (0-9):\n")
        f.write(" ".join(map(str, predictions)) + "\n\n")
        f.write("True Labels (0-9):\n")
        f.write(" ".join(map(str, true_labels)) + "\n\n")
        
        #basic accuracy
        correct = sum(p == l for p, l in zip(predictions, true_labels))
        accuracy = correct / len(predictions)
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})\n")
    
    print(f"CM data saved: {cm_file}")

def train_all_graph_types(time_bin="25ms"):  #<--------------------------------------------------------------set time bin
    
    """
    Reverted to August 17 working configuration
    This produced the good CSV logs: val_loss 2.29→2.00
    """
    
    #set seed=0 for reproducibility
    set_seed(0)
    print("Seed set to 0 for reproducible results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    #create log directory with time_bin
    log_dir = Path(f"training_logs_{time_bin}_{timestamp}")
    log_dir.mkdir(exist_ok=True)
    
    #transform configuration
    transform = Compose([RandomJitter((0.003, 0.003)), Cartesian(cat=False, norm=True)])
    
    graph_types = ['dir_spike_count_w', 'dir_spike_count', 'undir_spike_count_w', 'undir_spike_count', 'dir_time'] #<--------------------------------------------------------- TOGGLE GRAPH TYPES
    
    #graph_types = ['dir_time']  #test single graph

    results = {}

    print(f"Starting training for {len(graph_types)} graph types using {time_bin} data")  

    for i, graph_type in enumerate(graph_types, 1):
        print(f"\n[{i}/{len(graph_types)}] Starting {graph_type} ({time_bin})")  #<----------------------------------------- TOGGLE MODEL      

        model = model2(input_features=3, graph_type=graph_type) # <---- attributes: spike count, x, y.
        
        #model = model2_5l(input_features=3, graph_type=graph_type) # <-- 5 LAYERS

        #model = model2_3l(input_features=3, graph_type=graph_type) # <-- 3 LAYERS

        #model = model2_2l(input_features=3, graph_type=graph_type) # <-- 2 LAYERS


        tm = TrainModel(
            f'{time_bin}_ready_datasets/split_{graph_type}',
            model, 
            experiment_name=f'model2_{time_bin}_{graph_type}_{timestamp}',
            desc=f'Model2 {time_bin} comparison - {graph_type}',
            lr=0.001,   #original value
            features='pol',
            batch=4,   #original value
            n_epochs=100,
            merge_test_val=False,
            transform=transform,
            loss_func=torch.nn.CrossEntropyLoss()
        )
        
        #check the actual data TrainModel loaded
        sample_data = tm.train_data[0]  #get first training sample
        print(f"ACTUAL data.x shape: {sample_data.x.shape}")
        print(f"ACTUAL data.x first row: {sample_data.x[0] if len(sample_data.x) > 0 else 'empty'}")
        if hasattr(sample_data, 'pos'):
            print(f"ACTUAL data.pos shape: {sample_data.pos.shape}")
        else:
            print("No pos attribute found")


        try:
            print(f"Training {graph_type}...")
            tm.train()
            test_loss = tm.test()
            
            #calculate accuracy if available
            accuracy = 0
            if hasattr(tm, 'predictions') and hasattr(tm, 'true_labels'):
                accuracy = accuracy_score(tm.true_labels, tm.predictions)
                
                #save confusion matrix construction data
                save_cm_data(tm.predictions, tm.true_labels, graph_type, test_loss, log_dir, timestamp)
            
            results[graph_type] = {'loss': test_loss, 'accuracy': accuracy}
            print(f"{graph_type} completed - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}")
        
        except Exception as e:
            print(f"Training failed for {graph_type}: {e}")
            results[graph_type] = {'error': str(e)}
    
    #simple results summary
    successful = [(k, v) for k, v in results.items() if 'error' not in v]
    successful.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\nFinal Results ({time_bin}):")  #show time_bin
    for i, (graph_type, metrics) in enumerate(successful, 1):
        print(f"  {i}. {graph_type}: Acc={metrics['accuracy']:.3f}, Loss={metrics['loss']:.4f}")
    
    #save txt summary with accuracy
    with open(log_dir / 'summary.txt', 'w') as f:
        f.write(f"Training Summary ({time_bin} data, seed=0)\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (graph_type, metrics) in enumerate(successful, 1):
            f.write(f"{i}. {graph_type}: {metrics['accuracy']:.3f} accuracy, {metrics['loss']:.4f} loss\n")
    
    #save JSON results
    summary = {
        "timestamp": timestamp,
        "time_bin": time_bin,
        "seed": 0,
        "results": results
    }
    
    with open(log_dir / f'results_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results and confusion matrix data saved to: {log_dir}")
    
    return results

if __name__ == "__main__":
    
    #train on 10ms data (default)
    #print("="*60)
    #print("TRAINING ON 10MS DATA (seed=0)")
    #print("="*60)
    #results_10ms = train_all_graph_types(time_bin="10ms")

    #train on 25ms data
    print("\n" + "="*60) 
    print("TRAINING ON 25MS DATA (seed=0)")
    print("="*60)
    results_100ms = train_all_graph_types(time_bin="25ms")
    
    #train on 50ms data
    #print("\n" + "="*60) 
    #print("TRAINING ON 50MS DATA (seed=0)")
    #print("="*60)
    #results_50ms = train_all_graph_types(time_bin="50ms")
    
    #train on 100ms data
    #print("\n" + "="*60) 
    #print("TRAINING ON 100MS DATA (seed=0)")
    #print("="*60)
    #results_100ms = train_all_graph_types(time_bin="100ms")