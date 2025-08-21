from __future__ import division
import os.path as osp
import glob
from pprint import pprint
import torch
import torch_geometric.transforms as T
import os
import json 

import torch.utils.data
from torch_geometric.data import Data, DataLoader, Dataset

import numpy as np

import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from collections.abc import Sequence

import os.path as osp
from pathlib import Path

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, knn_graph
from torch_geometric.utils import remove_isolated_nodes
import math
import json

im_height=260
im_width=346

def files_exist(files):
    return all([osp.exists(f) for f in files])


class TactileDataset(Dataset):
    """TactileDataset for graph neural network training"""

    def __init__(
        self, 
        root, 
        transform=None, 
        pre_transform=None, 
        features='all', 
        temporal_edge_order=False, 
        reset=False, 
        augment=False
        ):
        """Initialize TactileDataset"""
        if reset:
            print('rm -rf ' + root + '/processed')
            ret=os.system('rm -rf ' + root + '/processed')
        root = Path(root).resolve()

        assert features in ['pol', 'coords', 'all', 'pol_time']
        self.features = features

        self.augment = augment
        self.temporal_edge_order = temporal_edge_order
        
        with open(root.parent / 'extraction_params.json', 'r') as f:
            self.params = json.load(f)
        
        print(self.params)
        possible_angle = self.params['possible_angles']
        N_examples = self.params['N_examples']
        list_of_rotations = [[0, 0, 0]]

        for i in range(1, N_examples):
            theta = i * 2 * math.pi/(N_examples - 1) if self.params['theta'] == 'full' else self.params['theta'][i]
            for phi in possible_angle:
                rx = phi * math.cos(theta)
                ry = phi * math.sin(theta)
                rotvec = [rx, ry, 0]
                list_of_rotations.append(rotvec)

        cases_dict = {i+1: list_of_rotations[i][:2] for i in range(len(list_of_rotations))}
        cases_dict[0] = [0, 0]
        self.cases_dict = cases_dict

        super(TactileDataset, self).__init__(root, transform, pre_transform)
        self._indices = None

    def indices(self) -> Sequence:
        return range(len(self)) if self._indices is None else self._indices

    @property
    def raw_file_names(self):
        filenames = os.path.join(self.raw_dir, 'contact_cases.json')
        file = [f.split('/')[-1] for f in filenames]
        return file

    @property
    def processed_file_names(self):
        filenames = glob.glob(str(self.root / 'processed' / 'sample_*.pt'))
        file = [f.split('/')[-1] for f in filenames]
        saved_file = [f.replace('.pt','.pt') for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        # Load the data file
        data = torch.load(self.root / 'processed' / f'sample_{idx+1}.pt', weights_only=False)
        
        # Remove problematic individual edge attributes that cause batching issues
        for attr in ['weight', 'time_edge']:
            if hasattr(data, attr):
                delattr(data, attr)
        
        # Ensure node time attribute - check for None properly
        if not hasattr(data, 'time') or data.time is None:
            data.time = torch.zeros(data.x.shape[0], dtype=torch.float)
        elif data.time.shape[0] != data.x.shape[0]:
            data.time = torch.zeros(data.x.shape[0], dtype=torch.float)
        
        # Ensure consistent edge_attr for all graphs
        num_edges = data.edge_index.size(1)
        if num_edges > 0:
            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                data.edge_attr = torch.ones((num_edges, 2), dtype=torch.float)
        else:
            data.edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        return data

    def sample_generator(self, samples_):
        for key, sample in samples_.items():
            case = sample['case']
            event_array = np.array(sample['events'])
            if not self.augment:
               yield case, event_array 
            else:
                for angle in [0, 90, 180, 270]:
                    if angle == 0:
                        yield case, event_array
                    else:
                        yield rotate_case(event_array, case, angle)

    def process(self):
        with open(self.root / 'raw' / 'contact_cases.json', 'r') as f:
            samples_ = json.load(f)

        samples = samples_
        if self.augment:
            samples = {}
            for i, (case, event_array) in enumerate(self.sample_generator(samples_)):
                samples[f'sample_{i+1}'] = {
                    'events': event_array,
                    'case': case
                }

        for sample_id in samples.keys():
            events = np.array(samples[sample_id]['events'])

            coord1, coord2 = torch.tensor(events[:, 0:2].astype(np.float32)).T 
            ts = events[:, 2]
            ts = ((ts - ts.min()) / (ts.max() - ts.min())).astype(np.float32)
            coord3 = torch.tensor(ts)
            pos = torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T

            if self.features == 'pol':
                feature = torch.tensor(2*events[:, 3].astype(np.float32)-1)
                feature = feature.view(-1, 1)
            elif self.features == 'coords':
                feature = torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T
            elif self.features == 'pol_time':
                feature = torch.stack((
                    torch.tensor(events[:, 3].astype(np.float32)),
                    coord3 
                )).T
            elif self.features == 'all':
                feature = torch.hstack((
                    torch.stack((coord1 / im_width, coord2 / im_height, coord3)).T, 
                    torch.tensor(events[:, 3].astype(np.float32)).reshape(-1, 1)
                    ))

            case = samples[sample_id]['case']

            edge_index = radius_graph(pos, r=0.05, max_num_neighbors=32, )

            if self.temporal_edge_order:
                row, col = edge_index
                temp_order_edges = pos[row, 2] <= pos[col, 2]
                edge_index = edge_index[:, temp_order_edges]

            if self.features == 'pol_time':
                pos = pos[:, :2]

            y = torch.tensor(np.array(self.cases_dict[case], dtype=np.float32)).reshape(1, -1)

            data = Data(x=feature, edge_index=edge_index, pos=pos, y=y)
            
            # Add time attribute for nodes
            data.time = torch.zeros(data.x.shape[0], dtype=torch.float)
            
            # Ensure consistent edge_attr
            num_edges = data.edge_index.size(1)
            if num_edges > 0:
                data.edge_attr = torch.ones((num_edges, 2), dtype=torch.float)
            else:
                data.edge_attr = torch.empty((0, 2), dtype=torch.float)
            
            transforms = T.Compose([T.Cartesian(cat=False, norm=True)])
            data = transforms(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                    continue

            if self.pre_transform is not None:
                    data = self.pre_transform(data)

            torch.save(data, self.root / 'processed' / f'{sample_id}.pt')

        self.params['node_features'] = self.features
        pprint(self.params)
        with open(self.root.parent / 'extraction_params.json', 'w') as f:
            json.dump(self.params, f, indent=4)

    def load_all_raw(self):
        samples = {}
        for subset in ['train', 'val', 'test']:
            with open(self.root.parent / subset / 'raw' / 'contact_cases.json', 'r') as f:
                subset_samples = json.load(f)
                subset_samples_tot_idx = {item[1]['total_idx']: 
                                        {
                                            'events': item[1]['events'], 
                                            'case': item[1]['case']
                                        } for item in subset_samples.items()}
            
            samples.update(subset_samples_tot_idx)
        return samples