import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATConv, LayerNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from copy import deepcopy
import numpy as np
import os
import argparse
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 1e-3
num_node_features = 33 # esm (33)
prediction_threshold = 0.5 # Default 0.5. if prediction probability is >=0.5, it is predicted as PSP

class GraphDataset(Dataset):

    def __init__(self, graph_list):
        '''
        graph_list is a numpy array of tuples defined as [(node_features, edge_index), (node_features, edge_index)]
        '''
        super().__init__()
        self.graph_list = graph_list 
        
    def len(self):
        return self.graph_list.shape[0]

    def get(self, idx):
       
        node_features_np, edge_index_np= self.graph_list[idx]
        
        t_node_features = torch.tensor(node_features_np, dtype=torch.float32)
        t_edge_index = torch.tensor(edge_index_np, dtype=torch.int64)

        data = Data(x=t_node_features, edge_index=t_edge_index)
        return data

class GINnet_bin(nn.Module):

    def __init__(self, dim_h):
        super(GINnet_bin, self).__init__()
        
        # GIN definition
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h), BatchNorm1d(dim_h), ReLU(), 
                       Linear(dim_h, dim_h), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), 
                       Linear(dim_h, dim_h), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),  
                       Linear(dim_h, dim_h), ReLU()))

        # classifier definition
        self.lin1 = Linear(dim_h*3, 50)
        self.lin2 = Linear(50, 20)
        self.lin3 = Linear(20, 1)

        self.sigmoid = nn.Sigmoid() # Sigmoid activation in the last layer

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        #h4 = self.conv3(h3, edge_index)

        # Graph-level readout
        #h0 = global_add_pool(x, batch)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        #h4 = global_add_pool(h4, batch)

        # Concatenate graph embeddings
        #h = h1
        #h = torch.cat((h1, h2), dim=1)
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin2(h)
        h = h.relu()
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin3(h)
        
        return self.sigmoid(h)

def prediction_bin(model, loader):
    model.eval()
    with torch.no_grad():
        all_pred = []
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred  = out.cpu().numpy()
            all_pred.append(pred)

        all_pred = np.array(all_pred, dtype=object)
        Y_PRED = np.concatenate(all_pred, axis=0)
        Y_PRED_PROB = deepcopy(Y_PRED)
        Y_PRED[Y_PRED >= prediction_threshold] = 1.0
        Y_PRED[Y_PRED < prediction_threshold] = 0.0
    
        # Reshape the arrays to 1D
        Y_PRED = Y_PRED.flatten()
        Y_PRED_PROB = Y_PRED_PROB.flatten()

    return Y_PRED_PROB

class GINnet_multiclass(nn.Module):

    def __init__(self, dim_h):
        super(GINnet_multiclass, self).__init__()
        
        # GIN definition
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h), BatchNorm1d(dim_h), ReLU(), 
                       Linear(dim_h, dim_h), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(), 
                       Linear(dim_h, dim_h), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),  
                       Linear(dim_h, dim_h), ReLU()))
        
        '''
        self.conv4 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),  
                       Linear(dim_h, dim_h), ReLU()))
        '''

        # classifier definition
        self.lin1 = Linear(dim_h*3, 50)
        self.lin2 = Linear(50, 20)
        self.lin3 = Linear(20, 7)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        #h4 = self.conv3(h3, edge_index)

        # Graph-level readout
        #h0 = global_add_pool(x, batch)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        #h4 = global_add_pool(h4, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin2(h)
        h = h.relu()
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.lin3(h)
        
        return h, F.log_softmax(h, dim=1)

def prediction_multiclass(model, loader):
    
    model.eval()
    pred = []
    scores = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            _, out = model(data.x, data.edge_index, data.batch)
        
        y_pred = out.argmax(dim=1).cpu().numpy()
        pred.append(y_pred)

        y = out.cpu().numpy()
        scores.append(y)

    Y_PRED = np.concatenate(pred, axis=0)
    # Reshape the arrays to 1D
    Y_PRED = Y_PRED.flatten()

    scores = np.concatenate(scores, axis = 0)
    return Y_PRED, scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-graphData", "--graphData", type = str, required = True, help = "Path to graph data (npy) file (required)")
    parser.add_argument("-bin_model", "--bin_model", type = str, required = True, help = "Path to binary model file (required)")
    parser.add_argument("-multiclass_model", "--multiclass_model", type = str, required = True, help = "Path to multi-class model file (required)")
    parser.add_argument("-idx_maps_id", "--idx_maps_id", type = str, required = True, help = "Index maps Protein ID json file (required)")
    parser.add_argument("-id_maps_prot_des", "--id_maps_prot_des", type = str, required = True, help = "Protein ID maps protein description json file (required)")
    parser.add_argument("-output_prefix", "--output_prefix", type = str, required = True, help = "Output prefix of the output csv file (required)")
    args = parser.parse_args()

    graphData_file = args.graphData
    bin_model_file = args.bin_model
    multiclass_model_file = args.multiclass_model
    idx_maps_id_json_file = args.idx_maps_id
    id_maps_prot_des_json_file = args.id_maps_prot_des
    output_prefix = args.output_prefix

    graphs = np.load(graphData_file, allow_pickle=True)
    dataset_for_prediction = GraphDataset(graphs)
    test_loader = DataLoader(dataset_for_prediction, batch_size=batch_size, shuffle=False)

    gin_model_bin = GINnet_bin(32)
    gin_model_bin.load_state_dict(torch.load(bin_model_file, map_location='cuda:0'))
    gin_model_bin.to(device)
    Y_PRED_PROB_bin = prediction_bin(gin_model_bin, test_loader)

    gin_model_multiclass = GINnet_multiclass(32)
    gin_model_multiclass.load_state_dict(torch.load(multiclass_model_file, map_location='cuda:0'))
    gin_model_multiclass.to(device)
    Y_PRED_multiclass, scores_multiclass = prediction_multiclass(gin_model_multiclass, test_loader)

    # Convert log softmax scores of multi-class classification to softmax scores
    softmax_scores = np.exp(scores_multiclass)
    # Normalize the scores to ensure they sum to 1
    softmax_scores = softmax_scores / softmax_scores.sum(axis=1, keepdims=True)

    with open(idx_maps_id_json_file) as json_file:
        IDX_maps_ID = json.load(json_file)
    
    with open(id_maps_prot_des_json_file) as json_file:
        ID_maps_Prot_Description = json.load(json_file)

    
    target_names = ["baseplate","major_capsid","major_tail","minor_capsid","minor_tail","portal","tail_fiber"]

    f = open(output_prefix+".csv", "a")
    f.write("Protein,PSP-Score,Binary-Label,baseplate,major_capsid,major_tail,minor_capsid,minor_tail,portal,tail_fiber,Multiclass-Label\n")

    
    for i in range(len(Y_PRED_PROB_bin)):
        
        idx = str(i)
        this_prot_ID = IDX_maps_ID[idx]
        this_prot_ID = str(this_prot_ID)
        this_prot_description = ID_maps_Prot_Description[this_prot_ID]

        f.write(this_prot_description+",")

        f.write(str(Y_PRED_PROB_bin[i])+",")

        if Y_PRED_PROB_bin[i] >= prediction_threshold:
            f.write("PSP,")
        else:
            f.write("non-PSP,")
        
        this_class_scores = softmax_scores[i]
        for j in this_class_scores:
            f.write(str(j)+",")
        
        f.write(target_names[Y_PRED_multiclass[i]] + "\n")
    
    f.close()