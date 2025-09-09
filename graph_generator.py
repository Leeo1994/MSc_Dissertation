# %%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as color
import matplotlib.cm as cmx
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import time, os, json
from itertools import combinations
import torch
from torch_geometric.utils import from_networkx

#led on-off switch
LED_STATUS = 'off'
bin_duration = 50  # ms

#base path
base_path = 'C:/Users/Teamaerosal/Desktop/Dissertation DUE 29TH AUG/Essentials for Braille Project/07_03_collect_10_row1_1/events/'

#loop through trials 0-9
for trial in range(10):
    print(f"\n{'='*70}")
    print(f"PROCESSING TRIAL {trial}, LED {LED_STATUS.upper()}")
    print(f"{'='*70}")
    
    file_path = f'{base_path}taps_trial_{trial}_pose_0_events_{LED_STATUS}.npy'
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
        
    '''Data loading'''
    data = np.load(file_path, allow_pickle = True)
    print(data.dtype.names)

    #check a single pixel:
    pixel_found = False
    for y in range(480):
        for x in range(640):
            if len(data[y][x]) > 0:
                print(f"First active pixel ({x}, {y}): {data[y][x]}")
                pixel_found = True
                break
        if pixel_found:
            break
    
    if not pixel_found:
        print("No active pixels found in this trial!")
        continue
        
    print("The data is an array containing lists. Each pixel has (x,y) values and a [timestamp]") 

    # %%
    #rearrange mapping to time based.

    indices = [i for i in np.ndindex(data.shape) if len(data[i]) > 0]

    #reverse the mapping:
    from collections import defaultdict

    #dictionary of timestamps + event indices for values
    organized_data = defaultdict(list)

    for i in indices:
        for ts in data[i]:
            organized_data[ts].append(i)

    #sorted list of timestamps + corresponding coordinates:
    sorted_timestamps = sorted(organized_data.keys())
    print(f"Found {len(sorted_timestamps)} unique timestamps")

    # %%
    import pandas as pd

    #flatten organized_data into list of dicts
    rows = []
    for ts, coords in organized_data.items():
        for x, y in coords:
            rows.append({'timestamp': ts, 'x': x, 'y': y})

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values('timestamp')
    print(f"Created DataFrame with {len(df_sorted)} total events")

    # %%
    '''================== TEMPORAL SEGMENTATION + GRAPH GENERATION =================='''
    
    #diagram generation functions:

    def get_braille_pattern(letter):
        """Get standard Braille pattern for letters A-J"""
        patterns = {
            'A': [(0, 0)], 'B': [(0, 0), (1, 0)], 'C': [(0, 0), (0, 1)], 
            'D': [(0, 0), (0, 1), (1, 1)], 'E': [(0, 0), (1, 1)], 
            'F': [(0, 0), (1, 0), (0, 1)], 'G': [(0, 0), (1, 0), (0, 1), (1, 1)], 
            'H': [(0, 0), (1, 0), (1, 1)], 'I': [(1, 0), (0, 1)], 'J': [(1, 0), (0, 1), (1, 1)]
        }
        return patterns.get(letter, [])

    def plot_braille_pattern(ax, letter):

        """Plot Braille pattern on given axis"""
        
        pattern = get_braille_pattern(letter)
        braille_grid = np.zeros((2, 2)) #2 rows for letters A-J --> change for extension
        for row, col in pattern:
            if row < 2:  #only use upper 2 rows
                braille_grid[row, col] = 1
        
        #show only upper 2 rows of circles
        for row in range(2):
            for col in range(2):
                circle = plt.Circle((col, row), 0.18,
                                  fill=braille_grid[row, col] > 0, 
                                  color='black' if braille_grid[row, col] > 0 else 'white',
                                  linewidth=1, edgecolor='lightgray')
                ax.add_patch(circle)
        
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)  
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_facecolor('white')
        ax.set_title(f'Braille Pattern', fontsize=18, fontweight='bold')  #slightly smaller font
        
    def create_complete_braille_analysis(segment_df, label, file_path, time_bins, event_counts, bin_duration):
        
        """Complete analysis showing temporal + spatial patterns"""
        
        data_dir = os.path.dirname(file_path)
        save_dir = os.path.join(data_dir, 'fixed_analysis')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        #set larger font sizes globally
        plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        })

        #create subplots with different widths - middle one smaller
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 3])  #middle subplot is 1/3 the width
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        #1. temporal pattern
        if len(event_counts) > 0:
            actual_bin_indices = [int(bin_start // bin_duration) for bin_start in sorted(time_bins.keys())]
            ax1.bar(actual_bin_indices, event_counts, width=1.0, edgecolor='black', color='steelblue', alpha=0.7)
            ax1.set_xlabel('Bin Index', fontsize=18)
            ax1.set_ylabel('Number of Spikes', fontsize=18)
            ax1.set_title(f'Temporal Pattern', fontsize=18, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=16)
        else:
            ax1.text(0.5, 0.5, 'No temporal data', ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            ax1.set_title(f'Temporal Pattern (Empty)', fontsize=18, fontweight='bold')
        
        #2. braille reference
        plot_braille_pattern(ax2, label)
        
        #3. spatial heatmap
        if len(segment_df) > 0:
            x_coords = segment_df['x'].values
            y_coords = segment_df['y'].values
            
            spatial_hist, _, _ = np.histogram2d(x_coords, y_coords, bins=[100, 75], range=[[0, 640], [0, 480]])
            
            im = ax3.imshow(spatial_hist.T, origin='lower', aspect='auto', 
                            extent=[0, 640, 0, 480], cmap='hot', interpolation='bilinear')
            
            ax3.set_xlabel('X Position', fontsize=18)
            ax3.set_ylabel('Y Position', fontsize=18)
            ax3.set_title(f'Sensor Activation', fontsize=18, fontweight='bold')
            ax3.tick_params(labelsize=16)
            
            cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
            cbar.set_label('Activation Intensity', fontsize=18)
            cbar.ax.tick_params(labelsize=14)
        else:
            ax3.text(0.5, 0.5, 'No spatial data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'Sensor Activation (Empty)', fontsize=18, fontweight='bold')
        
        main_title = f'Temporal and Spatial Input Analysis - Braille Letter {label}\n({bin_duration}ms time bins)'
        plt.suptitle(main_title, fontsize=22, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        save_filename = f'led_{LED_STATUS}_trial_{trial}_{bin_duration}ms_letter_{label}.png'
        plt.savefig(os.path.join(save_dir, save_filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        n_events = np.sum(event_counts) if len(event_counts) > 0 else 0
        print(f"Letter {label}: {len(event_counts)} bins, {n_events} total events, {len(segment_df)} data points")

    # %%
    import networkx as nx

    '''================== FIXED SEGMENTATION INTO EXACTLY 10 LETTERS =================='''

    graph_by_letter = {}

    #TEMPORAL SEGMENTATION
    print("=== FIXED TEMPORAL SEGMENTATION ===")
    print(f"Total data points: {len(df_sorted)}")

    total_time = df_sorted['timestamp'].max() - df_sorted['timestamp'].min()
    time_per_letter = total_time / 10  #10 letters A-J
    start_time = df_sorted['timestamp'].min()

    print(f"Total time span: {total_time:.0f}ms")
    print(f"Time per letter: {time_per_letter:.0f}ms")

    segments = []
    for i in range(10):
        segment_start = start_time + i * time_per_letter
        segment_end = start_time + (i + 1) * time_per_letter
        
        segment_data = df_sorted[
            (df_sorted['timestamp'] >= segment_start) & 
            (df_sorted['timestamp'] < segment_end)
        ].copy()
        
        segments.append(segment_data)
        print(f"Letter {i+1}: {len(segment_data)} points, time {segment_start:.0f}-{segment_end:.0f}ms")

    print(f"Created {len(segments)} segments (should be 10)")

    '''================= LABEL LETTERS =================='''

    braille_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    labels = braille_labels[:len(segments)]

    '''=================== BIN EACH LETTER INDIVIDUALLY =================='''

    for i, segment_df in enumerate(segments):
        label = labels[i] if i < len(labels) else f"Letter_{i}"
        print(f"\n===== Letter {label} =====")
        
        #skip empty segments
        if len(segment_df) == 0:
            print("Empty segment, skipping temporal analysis...")
            #still create empty entry in graph_by_letter for consistency
            graph_by_letter[label] = {
                "graph": nx.Graph(),
                "bins": {},
                "event_counts": np.array([]),
                "bin_duration": bin_duration
            }
            continue

        time_bins = defaultdict(list)
        for _, row in segment_df.iterrows():
            bin_start = (row['timestamp'] // bin_duration) * bin_duration
            time_bins[bin_start].append({
                'timestamp': row['timestamp'],
                'x': row['x'],
                'y': row['y']
            })

        #show first 5 time bins for each letter
        for j, (bin_start, coords_list) in enumerate(sorted(time_bins.items())):
            print(f"Bin starting at {bin_start}ms: {len(coords_list)} coordinates")
            print(f"  Sample coords: {coords_list[:3]}")
            if j >= 4:
                break

        #extract features AND spatial centroids from binned data
        node_features = []
        spatial_centroids = []

        for bin_start, events in sorted(time_bins.items()):
            event_count = len(events)
            
            #calculate spatial centroid for this time bin
            if events:
                centroid_x = np.mean([e['x'] for e in events])
                centroid_y = np.mean([e['y'] for e in events])
            else:
                centroid_x = centroid_y = 0.0
            
            node_features.append(event_count)
            spatial_centroids.append([centroid_x, centroid_y])

        event_counts = np.array(node_features)
        spatial_centroids = np.array(spatial_centroids)
        node_indices = np.arange(len(event_counts))

        #create graph with spatial node features
        G = nx.Graph()
        for k, (count, centroid) in enumerate(zip(event_counts, spatial_centroids)):
            G.add_node(k,
                    spike_count=count,
                    centroid_x=centroid[0],
                    centroid_y=centroid[1],
                    x=[count, centroid[0], centroid[1]],  #3D features for model2
                    pos=[centroid[0], centroid[1]])       #2D position for pooling

        graph_by_letter[label] = {
            "graph": G,
            "bins": time_bins,
            "event_counts": event_counts,
            "spatial_centroids": spatial_centroids,
            "bin_duration": bin_duration
        }

        '''==================== FIXED LETTER ANALYSIS ===================='''
        
        create_complete_braille_analysis(segment_df, label, file_path, time_bins, event_counts, bin_duration)

    print(f"\n=== SEGMENTATION SUMMARY ===")
    print(f"Successfully created {len(graph_by_letter)} letter segments:")
    for label, data in graph_by_letter.items():
        n_bins = len(data['event_counts'])
        n_events = np.sum(data['event_counts'])
        print(f"  {label}: {n_bins} time bins, {n_events} total events")

    print(f"Analysis plots saved to: {os.path.join(os.path.dirname(file_path), 'fixed_analysis')}")

    # %%
    '''=========================== UNWEIGHTED ==========================='''

    #-------------------------- SPIKE-COUNT BASED, DIRECTED ------------------------
    def DIR_SPIKE_COUNT(event_counts, base_graph, time_bins, bin_duration):

      '''Edges connect bins with same or increasing spike count - WEIGHTED - shorter edges for for identical spike counts'''

      title = "Directed Spike Activity-Ordered Graph"
      
      #convert to a directed graph with same nodes
      G = nx.DiGraph()
      G.add_nodes_from(base_graph.nodes(data=True))  # FIXED: preserve node attributes

      #add directed edges for non decreasing spike counts
      for i in range(len(event_counts)):
        for j in range(i + 1, len(event_counts)):
          if event_counts[i] <= event_counts[j]: 
            G.add_edge(i, j)

      return G, title, False #unweighted

    #---------------------------- SPIKE-COUNT BASED, UNDIRECTED -------------------------
    def UNDIR_SPIKE_COUNT(event_counts, base_graph, time_bins, bin_duration):

      '''Edges connect bins with identical spike counts'''

      #Adjacent - bins with identical spike counts
      title = "Spike Activity-Matched Undirected Graph"

      G = base_graph.copy() #reuse existing nodes

      for i in range(len(event_counts)):
        for j in range(i + 1, len(event_counts)):
          if event_counts[i] == event_counts[j]:
              G.add_edge(i, j, weight = 0.5) #identical spike counts

      return G, title, False #not weighted

    #---------------------------- TIME BASED, DIRECTED -----------------------------
    def DIR_TIME(event_counts, base_graph, time_bins, bin_duration):
        
      '''Edges connect time bins at a chronological order'''

      title = "Chronologically Ordered Time-Bin Graph"
        
        #convert to a directed graph with same nodes
      G = nx.DiGraph()
      G.add_nodes_from(base_graph.nodes(data=True))  # FIXED: preserve node attributes

        #connect each time bin to next one in sequence
      for i in range(len(time_bins) - 1):
          G.add_edge(i, i + 1, weight = 1)

      return G, title, False #not weighted

    '''============================ WEIGHTED ============================='''

    #-------------------------- SPIKE-COUNT BASED, UNDIRECTED ------------------------
    def UNDIR_SPIKE_COUNT_W(event_counts, base_graph, time_bins, bin_duration):

      '''Edges connect bins with identical spike counts - the more common a spike count is across the bins - the higher its edge's weight.'''

      #Adjacent - identical spike counts
      #Greater weight - the more identical counts the larger the weight 

      title = "Frequency-Weighted Equal Spike Activity Graph"
      weighted = True

      G = base_graph.copy() #reuse existing nodes

      matching_pairs = [
      (i,j)
      for i, j in combinations(range(len(event_counts)), 2)
        if event_counts[i] == event_counts[j]
      ]

      #total number of matching pairs
      total_matches = len(matching_pairs)

      if total_matches > 0:
        weight = 1 / total_matches #minimizing weight for shorter edges
      else: 
           weight = float('inf') #ignore it (really large number)

      for i, j in matching_pairs:
           G.add_edge(i, j, weight = weight) 

      return G, title, True #weighted

    #-------------------------- SPIKE-COUNT BASED, DIRECTED ------------------------
    def DIR_SPIKE_COUNT_W(event_counts, base_graph, time_bins, bin_duration):

      '''Edges connect bins with increasing spike count - WEIGHTED - shorter edges for for identical spike counts'''

      #Adjacent - bins with increasing spike counts + weights for identical counts
      title = "Frequency-Weighted Spike Activity-Ordered Graph"
      
      #convert to a directed graph with same nodes
      G = nx.DiGraph()
      G.add_nodes_from(base_graph.nodes(data=True))  # FIXED: preserve node attributes

      #find matching pairs (identical spike counts) for weighting
      matching_pairs = [
        (i, j)
        for i, j in combinations(range(len(event_counts)), 2)
        if event_counts[i] == event_counts[j]
      ]

      #calculate weight based on frequency
      total_matches = len(matching_pairs)
      if total_matches > 0:
        weight = 1 / total_matches
      else:
        weight = 1.0

      #add directed edges for increasing spike counts
      for i in range(len(event_counts)):
        for j in range(i + 1, len(event_counts)):
          if event_counts[i] <= event_counts[j]:
            edge_weight = weight if event_counts[i] == event_counts[j] else 1.0
            G.add_edge(i, j, weight = edge_weight)

      return G, title, True #weighted

    # %%
    states = {
        'dir_spike_count': DIR_SPIKE_COUNT,
        'undir_spike_count': UNDIR_SPIKE_COUNT,
        'dir_time': DIR_TIME,
        'undir_spike_count_w': UNDIR_SPIKE_COUNT_W,
        'dir_spike_count_w': DIR_SPIKE_COUNT_W
    }

    def graph_gen(G, title="Graph", weighted=False):
        """Generate and display graph using actual spatial positions"""
        plt.figure(figsize=(12, 8))
        
        #extract spatial coordinates from node data
        pos = {}
        for node in G.nodes():
            if 'centroid_x' in G.nodes[node] and 'centroid_y' in G.nodes[node]:
                x = G.nodes[node]['centroid_x'] 
                y = G.nodes[node]['centroid_y']
                pos[node] = (x, y)
            else:
                pos[node] = (float(node), 0.0)
        
        #color nodes by spike activity  
        node_colors = []
        for node in G.nodes():
            spike_count = G.nodes[node].get('spike_count', 1.0)
            node_colors.append(spike_count)
        
        #draw the graph with spatial positioning
        if weighted and G.number_of_edges() > 0:
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw(G, pos, with_labels=True, node_color=node_colors,
                    node_size=500, font_size=10, font_weight='bold',
                    edge_color='gray', width=[w*3 for w in weights], cmap='viridis')
        else:
            nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                    node_size=500, font_size=10, font_weight='bold',
                    edge_color='gray', cmap='viridis')
        
        #fix the colorbar creation
        if node_colors and len(set(node_colors)) > 1:  #only if colors vary
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                       norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
            sm.set_array([])  #empty array, not node_colors
            plt.colorbar(sm, ax=plt.gca(), label='Spike Count')  #specify current axes
        
        plt.xlabel('X Position (sensor pixels)')
        plt.ylabel('Y Position (sensor pixels)') 
        plt.title(title)
        #plt.show()  #COMMENTED OUT <--------------------------------------------------------------------- UNCOMMENT for graph pop-up windows
        plt.close()    #closes the figure to free memory
        
    def nx2pyg(G, label=None):
        """Convert NetworkX graph to PyTorch Geometric format - COMPLETE FIX"""
        import torch
        from torch_geometric.data import Data
        
        #add node features (x) and positions (pos) if not present
        for node in G.nodes():
            if 'x' not in G.nodes[node]:
                G.nodes[node]['x'] = 1.0  #default feature
            if 'pos' not in G.nodes[node]:
                G.nodes[node]['pos'] = [float(node), 0.0]  #default position

        #create data object manually instead of using from_networkx
        
        #create node features matrix (2D)
        node_features = []
        for node in sorted(G.nodes()):
            feat = G.nodes[node]['x']
            if isinstance(feat, (list, tuple)):
                node_features.append(feat)
            else:
                node_features.append([feat])  #wrap scalar in list to make 2D
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        #create positions matrix (2D)
        positions = []
        for node in sorted(G.nodes()):
            pos = G.nodes[node]['pos']
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                positions.append(pos[:2])  #take first 2 dimensions
            elif isinstance(pos, (list, tuple)) and len(pos) == 1:
                positions.append([pos[0], 0.0])  # Add y=0
            else:
                positions.append([float(pos), 0.0])  #handle scalar
        
        pos = torch.tensor(positions, dtype=torch.float)
        
        #create edge index matrix (2D)
        if G.number_of_edges() > 0:
            edge_list = []
            edge_weights = []
            edge_types = []
            
            #create mapping from node IDs to indices
            node_to_idx = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
            
            for u, v, edge_data in G.edges(data=True):
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                
                #add edge in both directions for undirected graphs
                if isinstance(G, nx.DiGraph):
                    edge_list.append([u_idx, v_idx])
                else:
                    edge_list.append([u_idx, v_idx])
                    edge_list.append([v_idx, u_idx])  #add reverse edge
                
                weight = edge_data.get('weight', 1.0)
                edge_weights.append(weight)
                edge_types.append(1.0)  #default edge type
                
                #add reverse edge attributes for undirected graphs
                if not isinstance(G, nx.DiGraph):
                    edge_weights.append(weight)
                    edge_types.append(1.0)
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[w, t] for w, t in zip(edge_weights, edge_types)], dtype=torch.float)
        else:
            #handle empty graphs
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        #create label tensor
        y = torch.tensor([label], dtype=torch.long) if label is not None else None
        
        #create data object directly with all components
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=y
        )
        
        return data

    # %%
    def generate_and_save_graphs(letter_dictionary, file_path, letter=None):
        
        '''
        generates graphs and converts to PyG format - saves for model2 with minimal changes
        '''

        print(f"DEBUG: Starting gen_and_save_graphs with file_path: {file_path}")
        print(f"DEBUG: letter_dictionary keys: {list(letter_dictionary.keys())}")
        
        #if specific letter requested, process only that letter, else all
        letters_to_process = [letter] if letter in letter_dictionary else letter_dictionary.keys()  
        print(f"DEBUG: letters_to_process: {list(letters_to_process)}")
          
        for letter_key in letters_to_process:
            data = letter_dictionary[letter_key]
            print(f"Processing graphs for letter {letter_key}...\n")
              
            event_counts = data['event_counts']
            time_bins = data['bins']
            bin_duration = data['bin_duration']
            base_graph = data['graph']

            #store generated graphs for model:
            data['graph_variants'] = {}  #for graph variants
            data['pyg_graphs'] = {}      #for pyg graphs
            data['graph_info'] = {}      #store title and weighted info for visualization

            for name, func in states.items():
                G, title, weighted = func(event_counts, base_graph, time_bins, bin_duration)
                print(f"Generated: {title}")

                data['graph_variants'][name] = G  #save graph in dictionary
                data['graph_info'][name] = {'title': title, 'weighted': weighted}  #save info for visualization
                
                # convert and store pyg graph
                pyg_graph = nx2pyg(G, label=None)  #use label if available
                data['pyg_graphs'][name] = pyg_graph

        print(f"DEBUG: Finished generating graphs. Starting save process")

        '''
        Save PyG graphs to individual .pt files for model training
        '''
        
        #create directories next to your .npy file
        data_dir = os.path.dirname(file_path)
        for s in ['train', 'val', 'test']:
            os.makedirs(os.path.join(data_dir, f'data/{s}/processed'), exist_ok=True)
        print(f"DEBUG: Directories created successfully")

        #extract info from filename
        file_base = os.path.basename(file_path)
        
        #extract trial number
        if 'trial_' in file_base:
            trial_num = file_base.split('trial_')[1].split('_')[0]
        else:
            trial_num = '1'  #default
        
        is_led_on = '_on.npy' in file_base
        led_status = 'on' if is_led_on else 'off'
        print(f"DEBUG: file_base = {file_base}")
        print(f"DEBUG: trial_num = {trial_num}, led_status = {led_status}")

        #file-based 70-15-15 split mapping
        train_trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  #14 files (70%)
        val_trials = [14, 15, 16]                                        #3 files (15%)
        test_trials = [17, 18, 19]                                       #3 files (15%)

        #determine split based on trial number
        trial_int = int(trial_num)
        if trial_int in train_trials:
            file_split = 'train'
        elif trial_int in val_trials:
            file_split = 'val'
        elif trial_int in test_trials:
            file_split = 'test'
        else:
            file_split = 'train'  #fallback for unexpected trial numbers

        print(f"DEBUG: trial_int = {trial_int}, assigned to split = {file_split}")

        print(f"DEBUG: Starting to save individual graphs...")
        
        graph_type_counters = {graph_type: 1 for graph_type in states.keys()}
        
        for letter_key, data in letter_dictionary.items():
            print(f"DEBUG: Processing letter_key: {letter_key}")
            if 'pyg_graphs' in data:
                print(f"DEBUG: Found pyg_graphs for {letter_key}, graph types: {list(data['pyg_graphs'].keys())}")
                for state_name, pyg_graph in data['pyg_graphs'].items():
                    print(f"DEBUG: Processing graph {state_name} for letter {letter_key}")
                    
                    #handle both single letters (A, B, C) and longer keys (Letter_10, etc.)
                    if len(letter_key) == 1 and letter_key.isalpha():
                        label_value = ord(letter_key) - ord('A')
                    else:
                        if letter_key.startswith('Letter_'):
                            label_value = int(letter_key.split('_')[1])
                        else:
                            label_value = 0
                    
                    print(f"DEBUG: letter_key='{letter_key}', label_value={label_value}")
                    pyg_graph.y = torch.tensor([label_value], dtype=torch.long)
                    
                    #use file-based split (all letters from this file go to same split)
                    split = file_split
                    
                    #save file
                    save_dir = os.path.join(data_dir, f'data/{split}/processed')
                    filename = f"trial_{trial_num}_events_{led_status}_{letter_key}_{state_name}.pt"
                    save_path = os.path.join(save_dir, filename)
                    print(f"DEBUG: Attempting to save to: {save_path}")
                    
                    torch.save(pyg_graph, save_path)
                    print(f"  ✓ Saved {state_name} graph for letter {letter_key} as {filename}")
                    graph_type_counters[state_name] += 1
            else:
                print(f"DEBUG: No pyg_graphs found for {letter_key}")
        
        #final print statements
        total_saved = sum(counter - 1 for counter in graph_type_counters.values())
        print(f"DEBUG: Finished saving. Total samples saved: {total_saved}")
        for graph_type, counter in graph_type_counters.items():
            print(f"Saved {counter-1} samples for {graph_type}")
        print(f"Files saved to: {os.path.join(data_dir, 'data/')}")
        print(f"Format: trial_{trial_num}_events_{led_status}_[LETTER]_[GRAPH_TYPE].pt")

    def visualize_graphs(letter_dictionary, letter = None):
        
        '''visualizes already generated graphs'''
        
        #if specific letter requested, process only that letter, else all
        letters_to_process = [letter] if letter in letter_dictionary else letter_dictionary.keys()  
          
        for letter_key in letters_to_process:
            data = letter_dictionary[letter_key]
            
            if 'graph_variants' in data and 'graph_info' in data:
                print(f"Visualizing graphs for letter {letter_key}...\n")
                for name in data['graph_variants'].keys():
                    G = data['graph_variants'][name]
                    title = data['graph_info'][name]['title']
                    weighted = data['graph_info'][name]['weighted']
                    
                    print(f"Visualizing: {title}")
                    graph_gen(G, title = f"{title} ({letter_key})", weighted = weighted)
            else:
                print(f"No graphs found for letter {letter_key}. Run generate_and_save_graphs() first.")

    # %%
    generate_and_save_graphs(graph_by_letter, file_path)

    # %%
    # isolate_data_object_issue()  #COMMENTED OUT <-- diagnostic function

    # %%
    # quick_graph_size_check()  #COMMENTED OUT <-- diagnostic function

    # %%
    # visualize_graphs(graph_by_letter)  #COMMENTED OUT <------------------------------------------------- uncomment for graph type visualization pop-up windows

    print(f"\n*** COMPLETED TRIAL {trial} ***")
    print(f"Processed {len(graph_by_letter)} letters: {list(graph_by_letter.keys())}")

print(f"\n{'='*50}")
print(f"BATCH PROCESSING COMPLETE")
print(f"LED Status: {LED_STATUS}")
print(f"Processed trials 0-9")
print(f"Graph types generated: {len(states)} expected")
print(f"States: {list(states.keys())}")
print(f"{'='*50}")