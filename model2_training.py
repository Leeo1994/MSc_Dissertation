import torch
from imports.TrainModel import TrainModel
from tuner import model2
from torch_geometric.transforms import RandomJitter, Compose, Cartesian
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from torch_geometric.nn import global_mean_pool

#============== cuda error workaround ==============
#force CPU mode
torch.cuda.is_available = lambda: False
os.environ['CUDA_VISIBLE_DEVICES'] = ''
#====================================================

def setup_logging():
    
    '''Create logging directory with timestamp'''
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"training_logs_{timestamp}")
    log_dir.mkdir(exist_ok=True)
    return log_dir, timestamp

def create_training_log_headers(log_dir, graph_type, timestamp):
    
    '''Create CSV file with headers for training logging'''
    
    csv_file = log_dir / f"training_log_{graph_type}_{timestamp}.csv"
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'max_loss', 'lr'])
    df.to_csv(csv_file, index=False)
    return csv_file

def log_epoch_results(csv_file, epoch, train_loss, val_loss, max_loss, lr):
    
    '''Append epoch results to CSV file'''
    
    new_row = pd.DataFrame([{
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'max_loss': max_loss,
        'lr': lr
    }])
    new_row.to_csv(csv_file, mode='a', header=False, index=False)

def save_results(results, log_dir, timestamp):
    
    '''Save final comparison results'''
    
    results_file = log_dir / f"results_{timestamp}.json"
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    summary = {
        "timestamp": timestamp,
        "results": results,
        "best_graph_type": sorted_results[0][0] if sorted_results else None,
        "best_test_loss": sorted_results[0][1] if sorted_results else None,
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return results_file

def override_pooling_size(model, new_size=512):
    """Override pooling layer sizes in the model"""
    
    print(f"🔧 Overriding pooling sizes to {new_size}")
    modified_count = 0
    
    for name, module in model.named_modules():
        # Look for pooling-related modules
        if 'pool' in name.lower() or 'Pool' in type(module).__name__:
            # Try common size attributes
            for attr in ['size', 'voxel_size', 'pool_size']:
                if hasattr(module, attr):
                    old_value = getattr(module, attr)
                    if isinstance(old_value, int) and old_value <= 64:
                        setattr(module, attr, new_size)
                        print(f"  Changed {name}.{attr}: {old_value} -> {new_size}")
                        modified_count += 1
                    elif isinstance(old_value, (list, tuple)) and any(x <= 64 for x in old_value):
                        # Handle cases where size is a list/tuple
                        new_value = [new_size if x <= 64 else x for x in old_value]
                        setattr(module, attr, new_value)
                        print(f"  Changed {name}.{attr}: {old_value} -> {new_value}")
                        modified_count += 1
    
    if modified_count == 0:
        print("  No pooling sizes found to modify")
    else:
        print(f"Modified {modified_count} pooling parameters")

def train_all_graph_types():
    
    '''Train models for each graph type'''
    
    log_dir, timestamp = setup_logging()
    
    #graph types to test
    #graph_types = ['dir_spike_count', 'dir_spike_count_w', 'undir_spike_count', 'dir_time', 'undir_spike_count_w']
    
    graph_types = ['dir_spike_count']  #individual graph testing:

    transform = Compose([RandomJitter((0.003, 0.003)), Cartesian(cat=False, norm=True)])
    results = {}
    
    #feature mapping for different graph types
    feature_mapping = {
        'dir_spike_count': 1,
        'dir_spike_count_w': 1,
        'undir_spike_count': 1,
        'dir_time': 1,
        'undir_spike_count_w': 1
    }

    for graph_type in graph_types:
        print(f"\nTraining {graph_type}")
        
        #create training log file
        csv_file = create_training_log_headers(log_dir, graph_type, timestamp)
        
        #get input features for current graph type
        input_features = feature_mapping.get(graph_type, 1)
        
        #create model
        model = model2(input_features=input_features, graph_type = graph_type)
        
        # *** POOLING SIZE OVERRIDE FOR LARGE GRAPHS ***
        override_pooling_size(model, 512)
        
        #setup training
        tm = TrainModel(
            f'data_{graph_type}',
            model, 
            lr=0.001, 
            features='pol',
            batch=4, 
            n_epochs=2,
            experiment_name=f'model2_{graph_type}_{timestamp}',
            desc=f'Model2 comparison - {graph_type}',
            merge_test_val=False,
            transform=transform,
            loss_func=torch.nn.CrossEntropyLoss()
        )
        
        #logging
        tm.csv_log_file = csv_file
        tm.log_epoch_callback = log_epoch_results
        
        #train and test
        try:
            tm.train()
            test_loss = tm.test()
            print(f"{graph_type} completed - Test Loss: {test_loss:.4f}")
        except Exception as e:
            print(f"Training failed for {graph_type}: {e}")
            test_loss = float('inf')
        
        results[graph_type] = test_loss
    
    #save results
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print("\nFinal Results:")
    for i, (graph_type, loss) in enumerate(sorted_results, 1):
        print(f"{i}. {graph_type}: {loss:.4f}")
    
    if sorted_results:
        print(f"\nBest: {sorted_results[0][0]} ({sorted_results[0][1]:.4f})")
    
    save_results(results, log_dir, timestamp)
    print(f"\nLogs saved to: {log_dir}")
    
    return results

if __name__ == "__main__":
    results = train_all_graph_types()


#%%

# %%
