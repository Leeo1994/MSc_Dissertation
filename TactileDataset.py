"""
TactileDataset.py - FINAL WORKING VERSION
Handles PyTorch Geometric Dataset initialization properly
"""
import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
import glob

class TactileDataset(Dataset):
    """
    Final working TactileDataset
    Sets up pt_files before parent __init__ to avoid property access issues
    """

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
        
        # Set up processed directory path manually
        root_path = Path(root)
        processed_path = root_path / 'processed'
        
        if not processed_path.exists():
            raise ValueError(f"Processed directory not found: {processed_path}")
        
        # Get all .pt files BEFORE calling super().__init__
        # This is needed because parent __init__ calls processed_file_names
        self.pt_files = sorted(list(processed_path.glob('*.pt')))
        
        if not self.pt_files:
            raise ValueError(f"No .pt files found in {processed_path}")
        
        # Now call parent __init__ - it can safely access processed_file_names
        super(TactileDataset, self).__init__(root, transform)

    def len(self):
        """Return dataset length"""
        return len(self.pt_files)

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

    @property
    def processed_file_names(self):
        """Return list of processed file names - called by parent __init__"""
        # This is called during parent __init__, so pt_files must be ready
        return [f.name for f in self.pt_files]

    @property
    def raw_file_names(self):
        """Return empty list - we work with processed files"""
        return []

    def process(self):
        """No processing needed - files are already processed"""
        pass

    def download(self):
        """No download needed"""
        pass