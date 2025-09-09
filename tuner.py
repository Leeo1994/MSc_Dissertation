import torch
from torch.nn import Linear
from torch.nn.functional import elu, dropout
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian
#CHANGED: Added new import for global pooling instead of complex custom pooling
from torch_geometric.nn import global_mean_pool
from models.modules import MaxPooling, MaxPoolingX
import numpy as np

class model2(torch.nn.Module):
    #CHANGED: Added comprehensive documentation about key features
    """
    key features: dim=2, global_mean_pool, 10 output classes
    """

    #CHANGED: Added input_features and graph_type parameters for configurable input
    def __init__(self, input_features=3, graph_type=None, pooling_size=(16/346, 12/260), pooling_outputs=32, pooling_after_conv2=False, more_layer=False, more_block=False):
        super(model2, self).__init__()
        

        '''
        NODE FEATURES = [spike_count, centroid_x, centroid_y] = 3D
        EACH ATTRIBUTE IS 2D - [WEIGHT, EDGE TYPE] = 2D
        POSITION DATA = [X,Y]  = 2D

        '''

        #CHANGED: Reduced from 3D to 2D spatial analysis for tactile sensor geometry
        dim = 2  
        bias = False
        root_weight = False
        kernel_size = 2

        self.pooling_size = np.array(pooling_size)
        self.pooling_output = pooling_outputs
        self.pooling_after_conv2 = pooling_after_conv2
        self.more_layer = more_layer
        self.more_block = more_block

        #CHANGED: Configurable input features instead of fixed input=1
        #network architecture
        self.conv1 = SplineConv(input_features, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=8)

        if self.more_layer:
            self.conv1_1 = SplineConv(8, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1_1 = BatchNorm(in_channels=8)

        self.conv2 = SplineConv(8, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=16)

        self.conv2_1 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2_1 = BatchNorm(in_channels=16)
        self.pool2_1 = MaxPooling(self.pooling_size/2, transform=Cartesian(norm=True, cat=False))

        self.conv3 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=16)

        self.conv4 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=16)

        self.conv5 = SplineConv(16, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=pooling_outputs)
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=pooling_outputs)
        
        self.conv7 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm7 = BatchNorm(in_channels=pooling_outputs)

        if self.more_block:
            self.conv6_1 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm6_1 = BatchNorm(in_channels=pooling_outputs)
            self.conv7_1 = SplineConv(pooling_outputs, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm7_1 = BatchNorm(in_channels=pooling_outputs)

        #CHANGED: Fixed output to 10 classes for Braille classification (A-J)
        #final layer - global pooling + 10 classes
        self.fc = Linear(pooling_outputs, out_features=10, bias=bias)

    def forward(self, data):
       
        #convolution layers
    
        #CHANGED: Added spatial coordinate normalization for 3D features
        if data.x.shape[1] >= 3:  #only if we have 3D features
            normalized_x = data.x.clone()
            #scale spatial coords: x_coord/640*100, y_coord/480*100 
            normalized_x[:, 1] = normalized_x[:, 1] / 640.0 * 100.0  # x coordinates
            normalized_x[:, 2] = normalized_x[:, 2] / 480.0 * 100.0  # y coordinates  
            data.x = normalized_x
    
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)

        if self.more_layer:
            data.x = elu(self.conv1_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1_1(data.x)
        
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        
        data.x = elu(self.conv2_1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2_1(data.x)

        if self.pooling_after_conv2:
            data = self.pool2_1(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)
        
        #residual connection
        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)

        if self.more_block:
            data.x = elu(self.conv6_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm6_1(data.x)

        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)

        if self.more_block:
            data.x = elu(self.conv7_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm7_1(data.x)

        #CHANGED: Replaced complex pooling with global mean pooling for robust classification
        #global pooling + classification
        x = global_mean_pool(data.x, data.batch)
        return self.fc(x)
    
#CHANGED: Added 5-layer variant of model2
class model2_5l(torch.nn.Module):
    
    """
    5-layer variant of model2 - removes conv6, conv7 layers
    Path: conv1 -> conv2 -> conv2_1 -> conv3 -> conv4 -> conv5 -> fc
    """

    def __init__(self, input_features=3, graph_type=None, pooling_size=(16/346, 12/260), pooling_outputs=32, pooling_after_conv2=False, more_layer=False, more_block=False):
        super(model2_5l, self).__init__()
        
        dim = 2
        bias = False
        root_weight = False
        kernel_size = 2

        self.pooling_size = np.array(pooling_size)
        self.pooling_output = pooling_outputs
        self.pooling_after_conv2 = pooling_after_conv2
        self.more_layer = more_layer

        #network architecture - stops at conv5
        self.conv1 = SplineConv(input_features, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=8)

        if self.more_layer:
            self.conv1_1 = SplineConv(8, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1_1 = BatchNorm(in_channels=8)

        self.conv2 = SplineConv(8, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=16)

        self.conv2_1 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2_1 = BatchNorm(in_channels=16)
        self.pool2_1 = MaxPooling(self.pooling_size/2, transform=Cartesian(norm=True, cat=False))

        self.conv3 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=16)

        self.conv4 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=16)

        self.conv5 = SplineConv(16, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=pooling_outputs)
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        #final layer - direct from conv5
        self.fc = Linear(pooling_outputs, out_features=10, bias=bias)

    def forward(self, data):
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)

        if self.more_layer:
            data.x = elu(self.conv1_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1_1(data.x)
        
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        
        data.x = elu(self.conv2_1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2_1(data.x)

        if self.pooling_after_conv2:
            data = self.pool2_1(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)
        
        #residual connection
        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        #global pooling + classification
        x = global_mean_pool(data.x, data.batch)
        return self.fc(x)
    

#CHANGED: Added 3-layer variant of model2
class model2_3l(torch.nn.Module):
    
    """
    3-layer variant of model2 - removes residual connections and pooling
    Path: conv1 -> conv2 -> conv3 -> fc
    """

    def __init__(self, input_features=3, graph_type=None, pooling_size=(16/346, 12/260), pooling_outputs=32, pooling_after_conv2=False, more_layer=False, more_block=False):
        super(model2_3l, self).__init__()
        
        dim = 2
        bias = False
        root_weight = False
        kernel_size = 2

        self.more_layer = more_layer

        #simple 3-layer architecture
        self.conv1 = SplineConv(input_features, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=8)

        if self.more_layer:
            self.conv1_1 = SplineConv(8, 8, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1_1 = BatchNorm(in_channels=8)

        self.conv2 = SplineConv(8, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=16)

        self.conv3 = SplineConv(16, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=pooling_outputs)

        #final layer
        self.fc = Linear(pooling_outputs, out_features=10, bias=bias)

    def forward(self, data):
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)

        if self.more_layer:
            data.x = elu(self.conv1_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1_1(data.x)
        
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)
        
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)

        #global pooling + classification
        x = global_mean_pool(data.x, data.batch)
        return self.fc(x)
    

#CHANGED: Added 2-layer variant of model2
class model2_2l(torch.nn.Module):
    
    """
    2-layer variant of model2 - minimal architecture
    Path: conv1 -> conv2 -> fc
    """

    def __init__(self, input_features=3, graph_type=None, pooling_size=(16/346, 12/260), pooling_outputs=32, pooling_after_conv2=False, more_layer=False, more_block=False):
        super(model2_2l, self).__init__()
        
        dim = 2
        bias = False
        root_weight = False
        kernel_size = 2

        self.more_layer = more_layer

        # Minimal 2-layer architecture
        self.conv1 = SplineConv(input_features, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=16)

        if self.more_layer:
            self.conv1_1 = SplineConv(16, 16, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1_1 = BatchNorm(in_channels=16)

        self.conv2 = SplineConv(16, pooling_outputs, dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=pooling_outputs)

        # Final layer
        self.fc = Linear(pooling_outputs, out_features=10, bias=bias)

    def forward(self, data):
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)

        if self.more_layer:
            data.x = elu(self.conv1_1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1_1(data.x)
        
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)

        #global pooling + classification
        x = global_mean_pool(data.x, data.batch)
        return self.fc(x)