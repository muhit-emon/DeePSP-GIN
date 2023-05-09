import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATConv, LayerNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from sklearn.metrics import classification_report
import numpy as np
import os

device = torch.device("cuda:1")
#device = "cpu"
batch_size = 128
learning_rate = 1e-3
num_node_features = 73
num_edge_features = 2
num_classes = 11


class GraphDataset(Dataset):

    def __init__(self, graph_list):
        '''
        graph_list is a numpy array of tuples defined as [(node_features, edge_index, class), (node_features, edge_index, class)]
        '''
        super().__init__()
        self.graph_list = graph_list 
        
    def len(self):
        return self.graph_list.shape[0]

    def get(self, idx):
       
        node_features_np, edge_index_np, labels_np = self.graph_list[idx] # w/o edge features
        #node_features_np, edge_index_np, edge_attr_np, labels_np = self.graph_list[idx] # with edge features
        
        
        labels_np = labels_np.reshape(-1, num_classes)
        #phanns_feats_np = phanns_feats_np.reshape(-1, phanns_feats_np.shape[0])

        t_node_features = torch.tensor(node_features_np, dtype=torch.float32)
        t_edge_index = torch.tensor(edge_index_np, dtype=torch.int64)
        #t_edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32) # if edge features included
        #t_phanns_feats = torch.tensor(phanns_feats_np, dtype=torch.float32)
        t_labels = torch.tensor(labels_np, dtype=torch.float32)

        data = Data(x=t_node_features, edge_index=t_edge_index, y=t_labels) # w/o edge features
        #data = Data(x=t_node_features, edge_index=t_edge_index, edge_attr=t_edge_attr, y=t_labels) # with edge features
        return data


class GINnet(nn.Module):

    def __init__(self, dim_h):
        super(GINnet, self).__init__()
        
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
        self.lin3 = Linear(20, num_classes)

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


class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, nheads=1, k=20):
        super(GATModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = nheads
        self.k = k

        # self.conv0 = GATConv(node_feature_dim, hidden_dim, heads=nheads)

        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=nheads)
        self.conv2 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads)
        self.conv3 = GATConv(nheads * hidden_dim, hidden_dim, heads=nheads, concat=False)

        # self.norm0 = LayerNorm(nheads * hidden_dim)
        self.norm1 = LayerNorm(nheads * hidden_dim)
        self.norm2 = LayerNorm(nheads * hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin0 = Linear(k * hidden_dim, hidden_dim)
        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        # edge_index, _  = dropout_adj(edge_index, p = 0.2, training = self.training)

        # x = self.conv0(x, edge_index)
        # x = self.norm0(x, batch)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)

        # 2. Readout layer
        #x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        
        x = self.topk_pool(x, edge_index, batch=batch)[0]
        x = x.view(batch[-1] + 1, -1)
        

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.lin0(x)
        x = F.relu(x)

        x = self.lin1(x)
        x = F.relu(x)

        z = x  # extract last layer features

        x = self.lin(x)

        return x, F.log_softmax(x, dim=1)

def test(model, loader):
    #criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    #total_loss = 0
    #acc = 0
    correct = 0
    pred = []
    true = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            _, out = model(data.x, data.edge_index, data.batch)
        
        #loss = F.nll_loss(out, data.y.argmax(dim=1))
        #total_loss += loss.item() * len(data)
        correct += (out.argmax(dim=1)==data.y.argmax(dim=1)).sum().item()
        y_pred = out.argmax(dim=1).cpu().numpy()
        y_true = data.y.argmax(dim=1).cpu().numpy()
        pred.append(y_pred)
        true.append(y_true)

    #total_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    #print("accuracy-->{:.2f}%".format(acc*100))
    Y_PRED = np.concatenate(pred, axis=0)
    Y_TRUE = np.concatenate(true, axis=0)

    target_names = ["baseplate","collar","HTJ","major_capsid","major_tail","minor_capsid","minor_tail","other","portal","shaft","tail_fiber"]
    report = classification_report(Y_TRUE, Y_PRED, target_names=target_names, output_dict=False)
    print(report)
    return



def test_ensemble(models, loader):
    #criterion = torch.nn.CrossEntropyLoss()
    for model in models:
        model.eval()
    #total_loss = 0
    #acc = 0
    correct = 0
    pred = []
    true = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            _, out1 = models[0](data.x, data.edge_index, data.batch)
            _, out2 = models[1](data.x, data.edge_index, data.batch)
            _, out3 = models[2](data.x, data.edge_index, data.batch)
            _, out4 = models[3](data.x, data.edge_index, data.batch)
            _, out5 = models[4](data.x, data.edge_index, data.batch)
            _, out6 = models[5](data.x, data.edge_index, data.batch)
        
        #loss = F.nll_loss(out, data.y.argmax(dim=1))
        #total_loss += loss.item() * len(data)

        out = torch.add(out1, out2)
        out = torch.add(out, out3)
        out = torch.add(out, out4)
        out = torch.add(out, out5)
        out = torch.add(out, out6)

        correct += (out.argmax(dim=1)==data.y.argmax(dim=1)).sum().item()
        y_pred = out.argmax(dim=1).cpu().numpy()
        y_true = data.y.argmax(dim=1).cpu().numpy()
        pred.append(y_pred)
        true.append(y_true)

    #total_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    #print("accuracy-->{:.2f}%".format(acc*100))
    Y_PRED = np.concatenate(pred, axis=0)
    Y_TRUE = np.concatenate(true, axis=0)

    target_names = ["baseplate","collar","HTJ","major_capsid","major_tail","minor_capsid","minor_tail","other","portal","shaft","tail_fiber"]
    report = classification_report(Y_TRUE, Y_PRED, target_names=target_names, output_dict=False)
    print(report)
    return


graphs_test = np.load("/home/muhitemon/graph_data_bin_wise_wo_edge_features/2D/2_graphData.npy", allow_pickle=True)
test_dataset = GraphDataset(graphs_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


'''
gin_model = GINnet(32)
best_model_path = "/home/muhitemon/saved_models_gin_wo_edge_features/test_11D/val_10_11D.pth"
gin_model.load_state_dict(torch.load(best_model_path))
gin_model.to(device)
test(gin_model, test_loader)
'''


#saved_models_path = "/home/muhitemon/saved_models_gin_wo_edge_features/test_11D" # saved gin models
saved_models_path = "/home/muhitemon/saved_models_gat_wo_edge_features/test_2D"  # saved gat models
best_models = os.listdir(saved_models_path)

saved_models = []
for model in best_models:
    if ".pth" in model:
        this_model_saved_state = os.path.join(saved_models_path, model)
        '''
        gin_model = GINnet(32)
        gin_model.load_state_dict(torch.load(this_model_saved_state, map_location=torch.device('cuda:1')))
        gin_model.to(device)
        saved_models.append(gin_model)
        '''
        
        gat_model = GATModel(num_node_features, 64, num_classes, 0.1)
        gat_model.load_state_dict(torch.load(this_model_saved_state, map_location=torch.device('cuda:1')))
        gat_model.to(device)
        saved_models.append(gat_model)
        
test_ensemble(saved_models, test_loader)
