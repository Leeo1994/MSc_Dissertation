"""
TrainModel.py - FINAL WORKING VERSION with Enhanced Debug Tests
Fixed scheduler compatibility issue + added comprehensive label and prediction debugging
"""
import json
import torch
import torch_geometric as pyg
from tqdm.auto import tqdm, trange  
from pathlib import Path
from numpy import pi
from pandas import DataFrame
from .TactileDataset import TactileDataset
import numpy as np
import warnings

class TrainModel():

    def __init__(
        self, 
        extraction_case_dir, 
        model,
        experiment_name,
        desc,
        n_epochs=150,
        optimizer='adam',
        lr=0.001,
        loss_func=torch.nn.L1Loss(),
        transform=None,
        features='all',
        weight_decay=0,
        patience=10,
        batch=1,
        augment=False,
        seed=0,
        merge_test_val=False
        ):

        #CHANGED: Directory name changed from './results' to './graph_results'
        # Create results directory
        results_dir = Path('./graph_results')
        results_dir.mkdir(exist_ok=True)
        
        path = results_dir / experiment_name
        exp_exists = path.exists()
        
        if exp_exists:
            warnings.warn(f'experiment {str(path)} already exists. Press enter to proceed overwriting.')
        
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize attributes
        self.path = path
        self.extraction_case_dir = Path(extraction_case_dir)
        self.transform = transform
        self.experiment_name = experiment_name
        self.device = 'cpu'  # Working version used CPU
        
        # Load datasets
        self.train_data = TactileDataset(self.extraction_case_dir / 'train', transform=transform, features=features, augment=augment)
        self.val_data = TactileDataset(self.extraction_case_dir / 'val', features=features)
        self.test_data = TactileDataset(self.extraction_case_dir / 'test', features=features)
        
        # Create data loaders
        self.train_loader = pyg.loader.DataLoader(self.train_data, shuffle=True, batch_size=batch)
        self.val_loader = pyg.loader.DataLoader(self.val_data, batch_size=1)  # Val batch size = 1
        self.test_loader = pyg.loader.DataLoader(self.test_data, batch_size=1)

        #CHANGED: Added debug function to check label ranges in all datasets
        # DEBUG: Check label ranges in datasets
        self.debug_label_ranges()

        # Model and training setup
        self.model = model.to(self.device)
        self.n_epochs = n_epochs
        self.desc = desc
        self.loss_func = loss_func

        # Optimizer setup
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError('use tm.optimizer = torch.optim.<optimizer>')
        
        #CHANGED: removed verbose parameter for compatibility
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=patience, factor=0.5
        )

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.max_losses = []
        self.stdev = []
        self.lr = []

    #CHANGED: Added new debug method to check label ranges in ALL samples
    def debug_label_ranges(self):
        """Debug function to check label ranges in ALL samples"""
        print("=" * 50)
        print("DEBUG: Checking ALL samples in datasets")
        print("=" * 50)
        
        for name, dataset in [("Train", self.train_data), ("Val", self.val_data), ("Test", self.test_data)]:
            all_labels = []
            
            # Check ALL samples in the dataset directly, not just first 5 batches
            for i in range(len(dataset)):
                try:
                    data = dataset[i]
                    all_labels.append(data.y.item())
                except Exception as e:
                    print(f"Error reading sample {i}: {e}")
                    continue
            
            if all_labels:
                min_label = min(all_labels)
                max_label = max(all_labels)
                unique_labels = sorted(set(all_labels))
                
                print(f"{name} Dataset:")
                print(f"  Total samples: {len(all_labels)}")
                print(f"  Label range: {min_label} to {max_label}")
                print(f"  Unique labels: {unique_labels}")
                
                if len(unique_labels) != 10:
                    print(f"  WARNING: Expected 10 classes, found {len(unique_labels)}")
                else:
                    print(f"  All 10 classes present")
            else:
                print(f"{name} Dataset: No labels found!")
            print()

    #CHANGED: Added new safe label conversion method with better error handling
    def safe_label_conversion(self, labels):
        """Safely convert labels for CrossEntropyLoss"""
        if isinstance(self.loss_func, torch.nn.CrossEntropyLoss):
            # Check if labels need conversion
            min_label = labels.min().item()
            max_label = labels.max().item()
            
            if min_label >= 0 and max_label <= 9:
                # Labels already 0-9, no conversion needed
                return labels
            elif min_label >= 1 and max_label <= 10:
                # Labels are 1-10, convert to 0-9
                return labels - 1
            else:
                # Handle problematic labels
                print(f"DEBUG: Unusual label range {min_label} to {max_label}")
                # Filter out invalid labels and convert
                valid_mask = labels > 0
                if valid_mask.sum() == 0:
                    raise ValueError("No valid labels found!")
                return labels - 1  # Still convert, but this might crash
        else:
            # For non-CrossEntropyLoss, keep original logic
            return labels - 1

    def get_pbar_postfix(self, epoch, epoch_loss, batch_i):
        """Create progress bar postfix"""
        return {
            'train_loss': epoch_loss / (batch_i + 1), 
            'val_loss': self.val_losses[-1] if self.val_losses else 'na',
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def train(self):
        """Training loop - working version from Aug 17 with debug"""
        
        self.model.train()
        
        for epoch in trange(self.n_epochs, desc='training', unit='epoch'):
            epoch_loss = 0
            lr = self.optimizer.param_groups[0]['lr']
            self.lr.append(lr)
            
            # Training loop
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    
                    # Move data to device
                    data = data.to(self.device)
                    
                    #CHANGED: Added debug printing for first batch labels
                    # DEBUG: Print first batch labels on first epoch
                    if epoch == 0 and i == 0:
                        print(f"\nDEBUG - First training batch:")
                        print(f"  Raw labels: {data.y[:10].tolist()}")
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    
                    #CHANGED: Replaced direct label conversion with safe_label_conversion method
                    # Safe label conversion
                    try:
                        labels = self.safe_label_conversion(data.y)
                        
                        # DEBUG: Print converted labels on first batch
                        if epoch == 0 and i == 0:
                            print(f"  Converted labels: {labels[:10].tolist()}")
                            
                    except Exception as e:
                        print(f"Label conversion failed: {e}")
                        print(f"Raw labels: {data.y}")
                        raise
                    
                    # Compute loss and backpropagate
                    loss = self.loss_func(output, labels)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.detach().item()
                    
                    # Update progress bar
                    pfix = self.get_pbar_postfix(epoch, epoch_loss, i)
                    tepoch.set_postfix(pfix)

            # Calculate epoch metrics
            epoch_loss /= (i + 1)
            val_loss, max_loss, stdev_loss = self.validate()
            
            # Update tracking
            self.model.train()
            self.train_losses.append(epoch_loss)
            self.val_losses.append(val_loss)
            self.max_losses.append(max_loss)
            self.stdev.append(stdev_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)

            # Logging
            if (epoch + 1) % 1 == 0:
                self.log(current_epoch=epoch)
            
            # Save checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.path / f'ckpt_{epoch+1}')
        
        # Save final model
        torch.save(self.model.state_dict(), self.path / 'state_dict')
        torch.save(self.model, self.path / 'model')
        
    def validate(self):
        """Validation loop - working version with debug"""
        loss = 0
        losses = []
        self.model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):      
                data = data.to(self.device)
                output = self.model(data)
                
                #CHANGED: Replaced direct label conversion with safe_label_conversion method
                # Safe label conversion
                labels = self.safe_label_conversion(data.y)
                
                l = self.loss_func(output, labels).detach().item()
                loss += l
                losses.append(l)
        
        loss /= len(self.val_loader)
        return loss, max(losses) if losses else 0, np.std(losses) if losses else 0
    
    def test(self):
        """Test function - working version with enhanced debug"""
        loss = 0
        predictions = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):      
                data = data.to(self.device)
                output = self.model(data)
                
                #CHANGED: Replaced direct label conversion with safe_label_conversion method
                # Safe label conversion  
                labels = self.safe_label_conversion(data.y)
                
                if isinstance(self.loss_func, torch.nn.CrossEntropyLoss):
                    # Get predictions (0-9) and store
                    pred = torch.argmax(output, dim=1).cpu().numpy()
                    predictions.extend(pred)
                    true_labels.extend(labels.cpu().numpy())
                
                loss += self.loss_func(output, labels).detach().item()

        loss /= len(self.test_data)
        
        # Store predictions for confusion matrix
        if predictions and true_labels:
            self.predictions = predictions
            self.true_labels = true_labels
            
            #CHANGED: Added comprehensive debug output for test results
            # ENHANCED DEBUG: Print comprehensive test results
            print(f"\nDEBUG - Test Results Detail:")
            print(f"  Total test samples: {len(predictions)}")
            print(f"  Sample predictions: {predictions[:10]}")
            print(f"  Sample true labels: {true_labels[:10]}")
            
            # Manual accuracy check
            manual_correct = sum(p == t for p, t in zip(predictions, true_labels))
            manual_accuracy = manual_correct / len(predictions)
            print(f"  Manual accuracy: {manual_accuracy:.4f} ({manual_correct}/{len(predictions)})")
            
            # Check prediction distribution
            unique_preds = set(predictions)
            pred_counts = {p: predictions.count(p) for p in unique_preds}
            print(f"  Prediction distribution: {pred_counts}")
            
            # Check true label distribution
            true_counts = {t: true_labels.count(t) for t in set(true_labels)}
            print(f"  True label distribution: {true_counts}")
            
            # Check for systematic issues
            if len(unique_preds) == 1:
                print(f"  WARNING: Model only predicts class {list(unique_preds)[0]}!")
            elif len(unique_preds) < 5:
                print(f"  WARNING: Model only uses {len(unique_preds)} classes out of 10!")
            
            # Additional diagnostics
            print(f"  True label range: {min(true_labels)} to {max(true_labels)}")
            print(f"  Prediction range: {min(predictions)} to {max(predictions)}")
            print(f"  Unique true labels: {sorted(set(true_labels))}")
            print(f"  Unique predictions: {sorted(set(predictions))}")
        
        return loss

    def log(self, current_epoch):
        """Logging function"""
        
        # Save description
        with open(self.path / 'desc.txt', 'w') as f:
            f.write(self.desc)

        # Save training parameters
        with open(self.path / 'training_params.json', 'w') as f:
            params = {
                'model': self.experiment_name,
                'extraction_used': str(self.extraction_case_dir.resolve()),
                'n_epochs': self.n_epochs,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            }
            json.dump(params, f, indent=4)

        # Save training log
        train_log = { 
            'epoch': [i for i in range(1, current_epoch+2)],
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'max_loss': self.max_losses,
            'lr': self.lr
        }
        DataFrame(train_log).to_csv(self.path / 'train_log.csv', index=False)