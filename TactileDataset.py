#CHANGED: Completely simplified docstring and removed complex parameter descriptions
"""
TactileDataset.py - FINAL WORKING VERSION
Handles PyTorch Geometric Dataset initialization properly
"""
#CHANGED: Drastically reduced imports - removed many dependencies
import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
import glob

class TactileDataset(Dataset):
    #CHANGED: Simplified class docstring
    """
    Final working TactileDataset
    Sets up pt_files before parent __init__ to avoid property access issues
    """

    #CHANGED: Completely simplified __init__ - removed many parameters
    def __init__(self, root, transform=None, features='all', augment=False):
        """
        Args:
            root: Path to data directory (train/val/test)  
            transform: Optional transform to apply
            features: Feature type (usually 'pol' or 'all')
            augment: Whether to apply augmentation
        """
        self.features = features
        self.augment = augment
        
        #CHANGED: Simplified path setup - no complex parameter loading
        # Set up processed directory path manually
        root_path = Path(root)
        processed_path = root_path / 'processed'
        
        if not processed_path.exists():
            raise ValueError(f"Processed directory not found: {processed_path}")
        
        #CHANGED: Pre-load pt_files before parent init to avoid property access issues
        # Get all .pt files BEFORE calling super().__init__
        # This is needed because parent __init__ calls processed_file_names
        self.pt_files = sorted(list(processed_path.glob('*.pt')))
        
        if not self.pt_files:
            raise ValueError(f"No .pt files found in {processed_path}")
        
        #CHANGED: Simplified super() call - removed pre_transform parameter
        # Now call parent __init__ - it can safely access processed_file_names
        super(TactileDataset, self).__init__(root, transform)

    #CHANGED: Simplified len() method name from __len__()
    def len(self):
        """Return dataset length"""
        return len(self.pt_files)

    #CHANGED: Completely rewritten get() method - much simpler file loading
    def get(self, idx):
        """Get data sample by index"""
        
        # Load the .pt file
        data_path = self.pt_files[idx]
        
        try:
            data = torch.load(data_path, map_location='cpu', weights_only=False)
            
            # Ensure data is a Data object
            if not isinstance(data, Data):
                raise ValueError(f"Expected Data object, got {type(data)}")
            
            # Apply transform if provided
            if self.transform is not None:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Error loading {data_path}: {e}")

    #CHANGED: Simplified processed_file_names - no complex glob patterns
    @property
    def processed_file_names(self):
        """Return list of processed file names - called by parent __init__"""
        # This is called during parent __init__, so pt_files must be ready
        return [f.name for f in self.pt_files]

    #CHANGED: Simplified raw_file_names - returns empty list instead of contact_cases.json
    @property
    def raw_file_names(self):
        """Return empty list - we work with processed files"""
        return []

    #CHANGED: Empty process() method - no data processing logic
    def process(self):
        """No processing needed - files are already processed"""
        pass

    #CHANGED: Added download() method that does nothing
    def download(self):
        """No download needed"""
        pass