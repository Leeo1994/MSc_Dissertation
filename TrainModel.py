"""
TrainModel.py - FINAL WORKING VERSION
Fixed scheduler compatibility issue
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

        # Create results directory
        results_dir = Path('./results')
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
        
        # Scheduler - FIXED: removed verbose parameter for compatibility
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=patience, factor=0.5
        )

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.max_losses = []
        self.stdev = []
        self.lr = []

    def get_pbar_postfix(self, epoch, epoch_loss, batch_i):
        """Create progress bar postfix"""
        return {
            'train_loss': epoch_loss / (batch_i + 1), 
            'val_loss': self.val_losses[-1] if self.val_losses else 'na',
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def train(self):
        """Training loop - working version from Aug 17"""
        
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
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    
                    # Label handling - convert 1-10 to 0-9 for CrossEntropyLoss
                    if isinstance(self.loss_func, torch.nn.CrossEntropyLoss):
                        labels = data.y - 1  # Convert 1-10 → 0-9 for PyTorch
                    else:
                        labels = data.y
                    
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
        """Validation loop - working version"""
        loss = 0
        losses = []
        self.model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):      
                data = data.to(self.device)
                output = self.model(data)
                
                # Same label handling as training
                if isinstance(self.loss_func, torch.nn.CrossEntropyLoss):
                    labels = data.y - 1  # Convert 1-10 → 0-9 for PyTorch  
                else:
                    labels = data.y
                
                l = self.loss_func(output, labels).detach().item()
                loss += l
                losses.append(l)
        
        loss /= len(self.val_loader)
        return loss, max(losses) if losses else 0, np.std(losses) if losses else 0
    
    def test(self):
        """Test function - working version"""
        loss = 0
        predictions = []
        true_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):      
                data = data.to(self.device)
                output = self.model(data)
                
                # Same label handling as training/validation
                if isinstance(self.loss_func, torch.nn.CrossEntropyLoss):
                    labels = data.y - 1  # Convert 1-10 → 0-9 for PyTorch
                    # Get predictions (0-9) and store
                    pred = torch.argmax(output, dim=1).cpu().numpy()
                    predictions.extend(pred)
                    true_labels.extend(labels.cpu().numpy())
                else:
                    labels = data.y
                
                loss += self.loss_func(output, labels).detach().item()

        loss /= len(self.test_data)
        
        # Store predictions for confusion matrix
        if predictions and true_labels:
            self.predictions = predictions
            self.true_labels = true_labels
        
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