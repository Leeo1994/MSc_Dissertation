import torch
import numpy as np
from imports.TrainModel import TrainModel
from tuner import model2
from torch_geometric.transforms import RandomJitter, Compose, Cartesian
import os
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score

# cuda error workaround
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.cuda.is_available = lambda: False

def train_all_graph_types():
    """
    Reverted to August 17 working configuration
    This produced the good CSV logs: val_loss 2.29→2.00
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Working transform configuration from Aug 17
    transform = Compose([RandomJitter((0.003, 0.003)), Cartesian(cat=False, norm=True)])
    
    # Graph types - same order as working version
    graph_types = ['dir_spike_count_w', 'dir_spike_count', 'undir_spike_count_w', 'undir_spike_count', 'dir_time']
    
    results = {}

    for i, graph_type in enumerate(graph_types, 1):
        print(f"\n[{i}/{len(graph_types)}] Starting {graph_type}")
        
        # Simple model creation - NO pooling override (this was the key bug)
        model = model2(input_features=1, graph_type=graph_type)
        
        # Working TrainModel configuration from Aug 17
        tm = TrainModel(
            f'data_{graph_type}',
            model, 
            experiment_name=f'model2_{graph_type}_{timestamp}',
            desc=f'Model2 comparison - {graph_type}',
            lr=0.001,  # Working learning rate from CSV logs
            features='pol',
            batch=4,   # Working batch size
            n_epochs=100,  # Working epoch count
            merge_test_val=False,
            transform=transform,
            loss_func=torch.nn.CrossEntropyLoss()
        )
        
        try:
            print(f"Training {graph_type}...")
            tm.train()
            test_loss = tm.test()
            
            # Calculate accuracy if available
            accuracy = 0
            if hasattr(tm, 'predictions') and hasattr(tm, 'true_labels'):
                accuracy = accuracy_score(tm.true_labels, tm.predictions)
            
            results[graph_type] = {'loss': test_loss, 'accuracy': accuracy}
            print(f"{graph_type} completed - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}")
        
        except Exception as e:
            print(f"Training failed for {graph_type}: {e}")
            results[graph_type] = {'error': str(e)}
    
    # Simple results summary
    successful = [(k, v) for k, v in results.items() if 'error' not in v]
    successful.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\nFinal Results:")
    for i, (graph_type, metrics) in enumerate(successful, 1):
        print(f"  {i}. {graph_type}: Acc={metrics['accuracy']:.3f}, Loss={metrics['loss']:.4f}")
    
    return results

if __name__ == "__main__":
    results = train_all_graph_types()